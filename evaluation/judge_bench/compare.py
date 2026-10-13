"""Compare frozen JudgeBench judgments and explain majority-class bias.

Reads saved predictions only; never runs a model or rewrites source artifacts.
Example: python -B -m evaluation.judge_bench.compare --input outputs/judge_bench_v1/test.jsonl \
    --before-run-dir outputs/judge_bench_v1/runs/qwen3.5-9b-retried \
    --output-dir outputs/judge_bench_v1/comparison
Add --after-run-dir after training and the identical 800-item evaluation finish.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .metrics import JOINT_THREE_FIELDS, compute_metrics, joint_label_metrics


MATCH_FIELDS = (
    "input_rows", "prompt_version", "prompt_sha256", "prompt_source_sha256", "schema_sha256",
    "max_tokens", "temperature", "top_p", "presence_penalty", "top_k",
    "thinking", "structured_output", "seed", "concurrency",
    "request_timeout_seconds", "max_retries", "scoring_policy",
)
LABELS = {"__null__": "null（不适用）", "__failed__": "失败／缺失"}
CORE = (("verdict", "Verdict"), ("timing", "Timing"),
        ("intensity", "Intensity（人工非空）"), ("joint_three", "Joint3（三项全对）"))


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _accuracy_delta_pp(before: dict, after: dict) -> float | None:
    """Compute the percentage-point difference from integer counts, avoiding ties lost to floats."""
    if not before["n"] or not after["n"]:
        return None
    left = Decimal(before["correct"]) / Decimal(before["n"])
    right = Decimal(after["correct"]) / Decimal(after["n"])
    return float((right - left) * Decimal(100))


def _read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def _valid_prediction(record: dict | None) -> dict | None:
    if record and record.get("status") == "ok" and isinstance(record.get("prediction"), dict):
        return record["prediction"]
    return None


def analyze(rows: list[dict], predictions: list[dict]) -> dict:
    """Compute label agreement and diagnostic populations without changing labels."""
    metrics = compute_metrics(rows, predictions, bootstrap_repeats=0)
    results = {record["item_id"]: record for record in predictions}
    grouped: dict[str, dict[str, list[dict]]] = {}
    for field in JOINT_THREE_FIELDS:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            value = row["gold"][field]
            groups["__null__" if value is None else str(value)].append(row)
        grouped[field] = groups
    default_pair = [row for row in rows if row["gold"]["timing"] == "Timely" and row["gold"]["intensity"] == "Appropriate"]
    non_default = [row for row in rows if not (row["gold"]["timing"] == "Timely" and row["gold"]["intensity"] == "Appropriate")]
    joint_groups = {
        "by_gold": {field: {label: joint_label_metrics(subset, results, JOINT_THREE_FIELDS)
                             for label, subset in sorted(groups.items())}
                    for field, groups in grouped.items()},
        "default_pair": joint_label_metrics(default_pair, results, JOINT_THREE_FIELDS),
        "non_default_pair": joint_label_metrics(non_default, results, JOINT_THREE_FIELDS),
    }
    errors = {field: Counter() for field in JOINT_THREE_FIELDS}
    for row in rows:
        prediction = _valid_prediction(results.get(row["item_id"]))
        for field in JOINT_THREE_FIELDS:
            gold = row["gold"][field]
            predicted = prediction.get(field) if prediction else "__failed__"
            if prediction is None or field not in prediction or gold != predicted:
                errors[field][("__null__" if gold is None else str(gold),
                               "__null__" if predicted is None else str(predicted))] += 1
    timing_correct = metrics["timing"]["correct"]
    intensity_correct = metrics["intensity"]["correct"]
    # Diagonals avoid deriving integer counts from floating-point recalls.
    def true_positive(field: str, label: str) -> int:
        matrix = metrics[field]["confusion"]
        return matrix["counts"][matrix["rows_gold"].index(label)][matrix["columns_prediction"].index(label)]
    timely = true_positive("timing", "Timely")
    appropriate = true_positive("intensity", "Appropriate")
    joint_correct = metrics["joint_three"]["correct"]
    sources = {
        "timing_correct_from_Timely": {"correct": timely, "total_correct": timing_correct,
                                       "fraction": timely / timing_correct if timing_correct else None},
        "intensity_correct_from_Appropriate": {"correct": appropriate, "total_correct": intensity_correct,
                                                "fraction": appropriate / intensity_correct if intensity_correct else None},
        "joint_three_correct_from_default_pair": {
            "correct": joint_groups["default_pair"]["correct"], "total_correct": joint_correct,
            "fraction": joint_groups["default_pair"]["correct"] / joint_correct if joint_correct else None,
        },
    }
    return {"metrics": metrics, "joint_three_groups": joint_groups, "correct_sources": sources,
            "major_errors": {field: [{"gold": gold, "prediction": prediction, "count": count}
                                     for (gold, prediction), count in counter.most_common()]
                             for field, counter in errors.items()}}


def paired_valid_analysis(rows: list[dict], before_predictions: list[dict], after_predictions: list[dict]) -> dict:
    """Diagnose the same successful-after IDs; never replace full-set headlines."""
    before = {record["item_id"]: record for record in before_predictions}
    after = {record["item_id"]: record for record in after_predictions}
    after_valid = {row["item_id"] for row in rows if _valid_prediction(after.get(row["item_id"])) is not None}
    selected = sorted(item_id for item_id in after_valid if _valid_prediction(before.get(item_id)) is not None)
    selected_set = set(selected)
    subset = [row for row in rows if row["item_id"] in selected_set]
    scores = {}
    for name, predictions in (("before", before_predictions), ("after", after_predictions)):
        metrics = compute_metrics(subset, predictions, bootstrap_repeats=0)
        scores[name] = {key: metrics[key] for key, _ in CORE}
    return {
        "selection_rule": "The same item IDs with structurally valid raw-content predictions after training and before training.",
        "interpretation": "Conditional diagnostic only: selection depends on the after-model output succeeding, so selection bias prevents treating these scores as full-test performance.",
        "items": len(selected), "after_valid_items": len(after_valid),
        "before_invalid_among_after_valid": len(after_valid) - len(selected),
        "excluded_from_full_test": len(rows) - len(selected),
        "item_ids": selected, "item_ids_sha256": _sha(_json(selected)),
        "item_ids_hash_encoding": "UTF-8 JSON array of sorted item IDs; compact separators, ensure_ascii=false",
        "denominators": {key: scores["before"][key]["n"] for key, _ in CORE},
        "before": scores["before"], "after": scores["after"],
        "accuracy_delta_percentage_points": {
            key: _accuracy_delta_pp(scores["before"][key], scores["after"][key])
            for key, _ in CORE
        },
        "primary_metrics_unchanged": True,
    }


def verify_format_diagnostics(diagnostics: dict, after: dict, after_predictions: list[dict], raw_sha256: str, artifact_sha256: str) -> dict:
    """Bind auxiliary channel evidence to the exact scored raw/prediction files."""
    source = diagnostics.get("source", {})
    if source.get("raw_responses", {}).get("sha256") != raw_sha256:
        raise ValueError("Format diagnostics raw response SHA-256 does not match the after run")
    prediction_sha256 = after["source_artifacts"]["predictions.jsonl"]["sha256"]
    if source.get("predictions", {}).get("sha256") != prediction_sha256:
        raise ValueError("Format diagnostics prediction SHA-256 does not match the after run")
    attempts = diagnostics.get("final_attempts", {})
    failures = sorted(record["item_id"] for record in after_predictions if _valid_prediction(record) is None)
    if (attempts.get("denominator") != after["metrics"]["total"]
            or attempts.get("failed_items") != after["metrics"]["failed_predictions"]
            or attempts.get("failed_item_ids") != failures):
        raise ValueError("Format diagnostics failed count/IDs differ from the scored after run")
    reasoning = diagnostics.get("reasoning_diagnostics", {})
    counters = [attempts.get("content_none_items"), *[reasoning.get(field) for field in (
        "whole_reasoning_valid_items", "embedded_structural_valid_items", "unambiguous_items", "ambiguous_items",
    )]]
    if any(type(value) is not int or not 0 <= value <= after["metrics"]["total"] for value in counters):
        raise ValueError("Format diagnostics counters must be nonnegative integer item counts")
    policy = diagnostics.get("policy", {})
    if policy.get("primary_scoring_unchanged") is not True or policy.get("auxiliary_only") is not True:
        raise ValueError("Format diagnostics must be auxiliary and preserve primary scoring")
    return {"verified": True, "raw_responses_sha256": raw_sha256, "predictions_sha256": prediction_sha256,
            "diagnostic_artifact_sha256": artifact_sha256, "failed_items": len(failures),
            "failed_item_ids_sha256": _sha(_json(failures)), "primary_metrics_unchanged": True}


def _load_run(directory: Path, rows: list[dict], input_sha256: str) -> tuple[dict, list[dict]]:
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    frozen = json.loads((directory / "frozen_prompt.json").read_text(encoding="utf-8"))
    predictions = _read_jsonl(directory / "predictions.jsonl")
    config = metadata["config"]
    if config.get("input_sha256") != input_sha256 or config.get("input_rows") != len(rows):
        raise ValueError(f"Run input hash/count differs from frozen input: {directory}")
    if metadata.get("complete") is not True or len(predictions) != len(rows):
        raise ValueError(f"Evaluation must finish all {len(rows)} items: {directory}")
    ids = [record["item_id"] for record in predictions]
    if len(set(ids)) != len(ids) or set(ids) != {row["item_id"] for row in rows}:
        raise ValueError(f"Run IDs do not match the frozen input exactly: {directory}")
    if config.get("prompt_sha256") != frozen.get("prompt_sha256") or config.get("schema_sha256") != _sha(_json(frozen["schema"])):
        raise ValueError(f"Frozen prompt/schema and run hashes disagree: {directory}")
    by_id = {record["item_id"]: record for record in predictions}
    for row in rows:
        record = by_id[row["item_id"]]
        if record.get("input_context_sha256") != _sha(_json(row["input_context"])) or not record.get("messages_sha256"):
            raise ValueError(f"Stored item context/message hash is missing or differs: {row['item_id']}")
    files = ("run_metadata.json", "frozen_prompt.json", "predictions.jsonl", "metrics.json")
    provenance = {name: {"sha256": _sha((directory / name).read_bytes()), "path": str((directory / name).resolve())}
                  for name in files if (directory / name).exists()}
    return {"run_directory": str(directory.resolve()), "config": config,
            "frozen_prompt": frozen, "source_artifacts": provenance, **analyze(rows, predictions)}, predictions


def check_match(before: dict, after: dict | None, before_predictions: list[dict], after_predictions: list[dict] | None) -> dict:
    """Model identity may change; evidence, instructions and inference settings may not."""
    if after is None:
        return {"status": "awaiting_after_run", "matched": None, "before_validated": True,
                "required_equal_fields": list(MATCH_FIELDS), "mismatches": [],
                "note": "训练后评测尚未提供；没有训练后分数或变化量。"}
    mismatches = [field for field in MATCH_FIELDS
                  if field not in before["config"] or field not in after["config"]
                  or before["config"][field] != after["config"][field]]
    for field in ("system_messages", "schema"):
        if before["frozen_prompt"].get(field) != after["frozen_prompt"].get(field):
            mismatches.append("frozen_prompt." + field)
    left = {record["item_id"]: record for record in before_predictions}
    right = {record["item_id"]: record for record in (after_predictions or [])}
    for item_id in sorted(left):
        for field in ("input_context_sha256", "messages_sha256"):
            if left[item_id].get(field) != right.get(item_id, {}).get(field):
                mismatches.append(f"{field}:{item_id}")
    return {"status": "matched" if not mismatches else "configuration_mismatch", "matched": not mismatches,
            "required_equal_fields": list(MATCH_FIELDS), "mismatches": mismatches,
            "before_settings": {field: before["config"].get(field) for field in MATCH_FIELDS},
            "after_settings": {field: after["config"].get(field) for field in MATCH_FIELDS},
            "same_item_evidence_and_messages": not any("sha256:" in field for field in mismatches),
            "model_identity_exception": "Model name/weights may change because of training; prompt/evidence/decoder settings must match.",
            "retry_note": "Different reproducible retry seeds may be selected under the saved first-structurally-valid policy; individual stochastic samples are not asserted identical."}


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    rounded = (Decimal(str(value)) * Decimal(100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{rounded:.2f}%"


def _pp(value: float | None) -> str:
    if value is None:
        return "—"
    rounded = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{rounded:+.2f} pp"


def _fraction(score: dict) -> str:
    return f"{score['correct']} / {score['n']} · {_pct(score['accuracy'])}"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{_e(x)}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{_e(x)}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def _confusion(score: dict) -> str:
    matrix = score["confusion"]
    return _table(["金标 ↓ / 预测 →", *[LABELS.get(label, label) for label in matrix["columns_prediction"]]],
                  [[LABELS.get(label, label), *counts] for label, counts in zip(matrix["rows_gold"], matrix["counts"])])


def render_html(comparison: dict, *, json_link: str = "comparison.json", artifact_links: list[tuple[str, str]] | None = None) -> str:
    before, after = comparison["before"], comparison["after"]
    intensity_n = before["metrics"]["intensity"]["n"]
    split = comparison.get("dataset_split")
    split_note = (f'<p class="note">数据划分：{_e(split["interpretation"])}</p>' if split else '')
    stages = [("训练前", before)] + ([("训练后", after)] if after else [])
    availability_cards = []
    for name, stage in stages:
        score = stage["metrics"]
        total, valid, failed = score["total"], score["valid_predictions"], score["failed_predictions"]
        availability_cards.append(
            f'<article><span>{_e(name)}输出可用性</span><strong>{valid} / {total} 结构有效</strong>'
            f'<span>{failed} / {total} 失败或缺失（{_pct(failed / total if total else None)}）</span></article>'
        )
    if after is None:
        availability_cards.append('<article><span>训练后输出可用性</span><strong>待训练和评测完成</strong><span>未提供结果，暂不计分</span></article>')
    availability = '<div class="availability">' + ''.join(availability_cards) + '</div>'
    failed_after = after["metrics"]["failed_predictions"] if after else 0
    availability_note = (
        f'<p class="availability-warning"><strong>训练后 {failed_after} / {after["metrics"]["total"]} 条失败或缺失。</strong>'
        '这些条目没有可计分的原始 content 标签，仍按固定分母计错。'
        '训练前后的主指标差值因此同时包含输出可用性变化，不能仅凭降幅判断语义标签能力下降。'
        '失败与有效输出内的分类错误应分别解释。</p>'
    ) if failed_after else ''
    diagnostics = comparison.get("format_diagnostics")
    channel_note = ""
    if diagnostics is not None:
        attempts = diagnostics["final_attempts"]
        reasoning = diagnostics["reasoning_diagnostics"]
        channel_note = (
            '<div class="availability-warning"><strong>已核验的输出通道诊断（不用于计分）</strong>'
            f'<p>训练后失败 {attempts["failed_items"]} 条；原始 content=None 的条目有 {attempts["content_none_items"]} 条。'
            f'reasoning 中检出完整、结构合法的判决 JSON 有 {reasoning["embedded_structural_valid_items"]} 条，'
            f'其中七字段对象唯一 {reasoning["unambiguous_items"]} 条、存在歧义 {reasoning["ambiguous_items"]} 条；'
            f'整个 reasoning 文本可直接作为合法 JSON 的有 {reasoning["whole_reasoning_valid_items"]} 条。'
            '因此有证据表明失败包含判决出现在其他输出通道的现象，不能把空 content 等同于模型没有作出标签判断。'
            '结构合法不代表标签与人工金标一致，唯一对象也不等于主指标中的有效 content。'
            '本报告不补提这些判决、不修复标签，四项主分数保持原始 content 计错口径。</p>'
            '<a href="format_diagnostics.json">查看独立 format_diagnostics.json（原始诊断）</a></div>'
        )
    conditional_section = ""
    cohort = comparison.get("paired_valid_subset")
    if cohort is not None:
        conditional_rows = [[title, _fraction(cohort["before"][key]), _fraction(cohort["after"][key]),
                             _pp(cohort["accuracy_delta_percentage_points"][key])]
                            for key, title in CORE]
        conditional_section = (
            '<section><h2>条件诊断：同一组训练后有效条目</h2>'
            f'<p>训练后结构有效 {cohort["after_valid_items"]} 条；其中训练前也有效的同一组 {cohort["items"]} 条用于下表。'
            f'与全部测试集相比排除 {cohort["excluded_from_full_test"]} 条。Intensity 仍仅计人工非空标签，'
            f'这一子集的分母是 {cohort["denominators"]["intensity"]}，主指标的分母是 {intensity_n}。</p>'
            + _table(["指标", "同一组：训练前", "同一组：训练后", "条件变化（百分点）"], conditional_rows)
            + '<p class="availability-warning"><strong>这是条件诊断，不是主结果。</strong>'
              f'条目按训练后的输出成功与否筛选，存在选择偏差；即使同一组标签准确率提高，也不能替代固定 800 / {intensity_n} 分母的主分数，'
              '不能据此宣称整个测试集上的净提升。它只用于区分可返回标签的条目表现与输出通道可用性变化。</p>'
            + f'<p class="note">同一组排序 item IDs 的 SHA-256：<code class="hash">{_e(cohort["item_ids_sha256"])}</code>。'
              '完整 IDs、分母与筛选规则保存在 comparison.json。</p></section>'
        )
    core_rows = []
    for key, title in CORE:
        score = before["metrics"][key]
        constant = before["metrics"]["baselines"]["always_majority"][key]
        delta = comparison["accuracy_delta_percentage_points"].get(key) if after else None
        core_rows.append([title, _fraction(score), _fraction(after["metrics"][key]) if after else "待训练和评测完成",
                          _pp(delta), _fraction(constant)])
    sources = before["correct_sources"]
    t, i, j = (sources[key] for key in ("timing_correct_from_Timely", "intensity_correct_from_Appropriate", "joint_three_correct_from_default_pair"))
    interpretation = (
        f"Timing 的 {t['total_correct']} 个正确结果中，{t['correct']} 个是 Timely（{_pct(t['fraction'])}）；"
        f"Intensity 非空条目的 {i['total_correct']} 个正确结果中，{i['correct']} 个是 Appropriate（{_pct(i['fraction'])}）。"
        f"Joint3 的 {j['total_correct']} 个正确结果中，{j['correct']} 个来自金标 Timely＋Appropriate（{_pct(j['fraction'])}）。"
        "结合各类别召回和非默认子集，才能判断总体准确率是否主要由多数类别支撑。"
    )
    reference_notes = []
    for key, title in (("timing", "Timing"), ("intensity", "Intensity")):
        observed = before["metrics"][key]["accuracy"]
        constant = before["metrics"]["baselines"]["always_majority"][key]
        relation = "高于" if observed > constant["accuracy"] else "低于" if observed < constant["accuracy"] else "等于"
        reference_notes.append(f"{title} 准确率{relation}常量 {constant['predicted_constant_label']} 的参考分数。")
    reference_note = "".join(reference_notes)
    class_sections = []
    for key, title in (("verdict", "Verdict"), ("timing", "Timing"), ("intensity", "Intensity：人工非空条目")):
        rows = []
        for stage_name, stage in stages:
            matrix = stage["metrics"][key]["confusion"]
            for label, score in stage["metrics"][key]["per_class"].items():
                tp = matrix["counts"][matrix["rows_gold"].index(label)][matrix["columns_prediction"].index(label)]
                rows.append([stage_name, label, score["support"], score["predicted"], tp,
                             f"{tp}/{score['predicted']} · {_pct(score['precision'])}" if score["predicted"] else "—（未预测该类）",
                             f"{tp}/{score['support']} · {_pct(score['recall'])}" if score["support"] else "—", _pct(score["f1"])])
        matrices = ''.join(f"<h4>{_e(stage_name)}混淆矩阵</h4>{_confusion(stage['metrics'][key])}" for stage_name, stage in stages)
        class_sections.append(f"<section><h2>{_e(title)}：类别与混淆矩阵</h2>" + _table(
            ["阶段", "类别", "金标条数", "预测条数", "正确", "Precision", "Recall", "F1"], rows,
        ) + matrices + "</section>")
    balanced_rows = []
    for key, title in (("verdict", "Verdict"), ("timing", "Timing"), ("intensity", "Intensity（非空）")):
        for name, stage in stages:
            score = stage["metrics"][key]
            balanced_rows.append([title, name, _fraction(score), _pct(score["macro_f1"]), _pct(score["balanced_accuracy"])])
        constant = before["metrics"]["baselines"]["always_majority"][key]
        balanced_rows.append([title, f"常量 {constant['predicted_constant_label']}", _fraction(constant), _pct(constant["macro_f1"]), _pct(constant["balanced_accuracy"])])
    subset_rows = []
    for key, title in (("timing_non_timely", "金标非 Timely"), ("intensity_non_appropriate", "金标 Excessive / Insufficient"),
                       ("intensity_null_cases", "金标 intensity=null"), ("intensity_including_null", "Intensity 全部条目（含 null）")):
        for name, stage in stages:
            score = stage["metrics"][key]
            subset_rows.append([title, name, _fraction(score), _pct(score["macro_f1"]), _pct(score["balanced_accuracy"])])
    joint_rows = []
    for name, stage in stages:
        groups = stage["joint_three_groups"]
        joint_rows.extend([[name, "Timely＋Appropriate", _fraction(groups["default_pair"])],
                           [name, "至少一个非默认标签，含 null", _fraction(groups["non_default_pair"])]])
        for field in ("verdict", "timing", "intensity"):
            joint_rows.extend([name, f"金标 {field}={LABELS.get(label, label)}", _fraction(score)]
                              for label, score in groups["by_gold"][field].items())
    error_rows = []
    for name, stage in stages:
        for field, errors in stage["major_errors"].items():
            error_rows.extend([name, field, LABELS.get(error["gold"], error["gold"]),
                               LABELS.get(error["prediction"], error["prediction"]), error["count"]]
                              for error in errors[:6])
    quality_rows = [[name, stage["metrics"]["valid_predictions"], stage["metrics"]["failed_predictions"],
                     _fraction(stage["metrics"]["joint_six"]), _fraction(stage["metrics"]["verdict_with_strict_consistency"])]
                    for name, stage in stages]
    config_rows = [[field, before["config"].get(field), after["config"].get(field) if after else "待评测"] for field in MATCH_FIELDS]
    links = ''.join(f'<a href="{_e(href)}">{_e(label)}</a>' for label, href in (artifact_links or []))
    status = "同输入、同 prompt、同推理设置，匹配检查通过" if after else "训练后结果待完成；当前仅展示训练前诊断"
    null_matrices = ''.join(f"<h4>{_e(name)}：全部 800 条，null 为独立类别</h4>{_confusion(stage['metrics']['intensity_including_null'])}" for name, stage in stages)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>JudgeBench · 训练前后与类别偏向</title>
<style>body{{margin:0;background:#f3f6f8;color:#173044;font:15px/1.65 system-ui,-apple-system,"PingFang SC",sans-serif}}main{{max-width:1180px;margin:auto;padding:32px 24px 60px}}h1{{font-size:30px;line-height:1.3}}h2{{font-size:21px}}h4{{margin:20px 0 8px}}section{{background:white;border:1px solid #dce5ec;border-radius:14px;padding:24px;margin:18px 0}}.note{{color:#5f7180;font-size:13px}}.hash{{overflow-wrap:anywhere}}.status{{background:#e7f2ef;padding:12px 16px;border-radius:10px}}.availability{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-bottom:16px}}.availability article{{padding:14px 16px;border:1px solid #dce5ec;border-radius:10px;background:#f5f8fa}}.availability span,.availability strong{{display:block}}.availability strong{{font-size:20px;margin:5px 0}}.availability-warning{{padding:14px 16px;background:#fff4df;border:1px solid #e9cd91;border-radius:10px}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:10px 12px;border-bottom:1px solid #dce5ec;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{background:#f5f8fa;font-size:13px}}td{{min-width:75px}}a{{color:#14665d}}.links{{display:flex;gap:14px;flex-wrap:wrap}}@media(max-width:600px){{main{{padding:20px 10px}}section{{padding:16px 12px}}h1{{font-size:25px}}.availability{{grid-template-columns:1fr}}}}@media print{{body{{background:white}}main{{padding:0}}section{{break-inside:avoid}}}}</style></head><body><main>
<header><h1>JudgeBench：训练前后与类别偏向</h1><p class="status">{_e(status)}</p><p>固定 800 条人工标注测试集；模型评价的是候选 proactive action 是否正确，以及时机、强度是否合适。</p>{split_note}</header>
<section><h2>四项核心结果</h2>{availability}{availability_note}{channel_note}{_table(['指标','训练前','训练后','变化（百分点）','金标多数类常量参考'], core_rows)}<p class="note">主指标仅计 API 原始 content 中结构有效的判决；解析失败、API 失败或缺失输出都按固定分母计错，不从其他字段补提判决。Verdict / Timing / Joint3 分母为全部 800 条。Intensity 主指标仅使用 {intensity_n} 条人工标签非空条目；Joint3 包含 null 并要求逐项精确匹配。辅助标签内部不一致不修正。常量参考由测试标签分布计算，不是另一模型的推理结果。</p></section>
{conditional_section}
<section><h2>训练前分数应如何理解</h2><p>{_e(interpretation)}</p><p>{_e(reference_note)} Macro F1 对各类 F1 等权；balanced accuracy 对各类召回等权，因此多数类不能单独支撑高分。</p><p>这些数字评价 judge 识别被评价行动错误的能力。人工金标中的 Timely、Appropriate 或 acceptable 占比描述候选行动及标注分布，不代表 judge 的能力；模型预测占比描述它的判断偏向。训练后需要同时观察异常类别召回、非默认子集与 Joint3。</p>{_table(['指标','模型／常量','正确 / 分母','Macro F1','Balanced accuracy'], balanced_rows)}</section>
{''.join(class_sections)}
<section><h2>非默认类别与 null 的表现</h2>{_table(['人工标签子集','阶段','正确 / 分母','Macro F1','Balanced accuracy'], subset_rows)}<p class="note">非 Appropriate 子集只含 Excessive / Insufficient；null 表示不适用，单列。Intensity 主混淆矩阵包含预测 null 列，但排除人工 null 行；所以上面的预测条数仅针对 {intensity_n} 条，不能与全 800 条分布混用。</p>{null_matrices}</section>
<section><h2>Joint3：三项同时正确的来源</h2>{_table(['阶段','人工标签分组','三项全对 / 分母'], joint_rows)}<p class="note">Joint3 要求 verdict、timing、intensity 同时与人工金标一致。它不是判决正确且诊断内部一致；后者可以在错误的辅助标签之间保持自洽。</p></section>
<section><h2>主要错误</h2>{_table(['阶段','指标','人工标签','错误预测','次数'], error_rows)}<p class="note">每项最多显示六种最常见错误转移；完整计数保存在 JSON。分类错误不等于 API 或 JSON 结构失败。</p></section>
<section><h2>输出质量与严格口径</h2>{_table(['阶段','结构有效','失败 / 缺失','六项金标全对','判决正确且内部一致'], quality_rows)}<p class="note">六项金标全对包含 action_intensity、rejection_dimensions（按集合）与 other_issue；rationale 文本不参与标签准确率评分。原始标签保持不变。</p></section>
<section><h2>可复现检查与原始文件</h2>{_table(['必须匹配的字段','训练前','训练后'], config_rows)}<p class="note">还检查测试文件 SHA-256、全部 item IDs、每条 context/messages SHA-256、冻结 system messages 与 schema。模型名和权重允许随训练改变；不允许用不同 thinking 或解码设置宣称匹配比较。保存的可复现重试 seed 可因首个结构有效输出而不同，单次随机样本不宣称相同。</p><p class="links"><a href="{_e(json_link)}">机器可读 comparison.json</a>{links}</p></section>
<footer class="note">从保存的 predictions 现场重算（bootstrap=0），不执行推理，不改写原始 metrics、prompt、人工标签或测试数据。未提供训练后结果时，不填充 0 分或估计值。</footer></main></body></html>'''


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--before-run-dir", type=Path, required=True)
    parser.add_argument("--after-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, help="Verified dataset scope and overlap evidence for this comparison")
    args = parser.parse_args(argv)
    input_bytes = args.input.read_bytes()
    rows = _read_jsonl(args.input)
    if len(rows) != 800 or len({row["item_id"] for row in rows}) != 800:
        raise ValueError("Comparison requires the frozen 800 unique JudgeBench items")
    input_sha256 = _sha(input_bytes)
    before, before_predictions = _load_run(args.before_run_dir, rows, input_sha256)
    after = after_predictions = None
    if args.after_run_dir:
        after, after_predictions = _load_run(args.after_run_dir, rows, input_sha256)
    match = check_match(before, after, before_predictions, after_predictions)
    if match["matched"] is False:
        raise ValueError("Cannot generate a matched comparison; differing fields: " + ", ".join(match["mismatches"]))
    output_dir = args.output_dir.resolve()
    diagnostics = diagnostic_verification = None
    diagnostic_path = output_dir / "format_diagnostics.json"
    if after is not None and diagnostic_path.exists():
        diagnostic_bytes = diagnostic_path.read_bytes()
        diagnostics = json.loads(diagnostic_bytes)
        raw_path = args.after_run_dir / "raw_responses.jsonl"
        raw_sha256 = _sha(raw_path.read_bytes())
        diagnostic_verification = verify_format_diagnostics(
            diagnostics, after, after_predictions, raw_sha256, _sha(diagnostic_bytes),
        )
        after["source_artifacts"]["raw_responses.jsonl"] = {"path": str(raw_path.resolve()), "sha256": raw_sha256}
    comparison = {
        "format_version": "judge-bench-comparison-v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(args.input.resolve()), "sha256": input_sha256, "items": len(rows)},
        "status": match["status"], "match_check": match, "before": before, "after": after,
        "accuracy_delta_percentage_points": {key: _accuracy_delta_pp(before["metrics"][key], after["metrics"][key]) for key, _ in CORE} if after else None,
        "paired_valid_subset": paired_valid_analysis(rows, before_predictions, after_predictions) if after else None,
        "format_diagnostics": diagnostics,
        "format_diagnostics_verification": diagnostic_verification,
        "policy": "Fresh metrics from stored predictions; all selected human labels must match. No source artifacts changed. After training is absent until a complete matched evaluation is supplied.",
    }
    if args.split_manifest:
        split_bytes = args.split_manifest.read_bytes()
        split = json.loads(split_bytes)
        if split["outputs"]["test"]["sha256"] != input_sha256 or split["outputs"]["test"]["items"] != len(rows):
            raise ValueError("Dataset manifest differs from the scored holdout")
        comparison["dataset_split"] = {
            "path": str(args.split_manifest.resolve()), "sha256": _sha(split_bytes),
            "benchmark_id": split["benchmark_id"], "selection": split["selection"],
            "overlap": split["overlap"], "interpretation": split["interpretation"],
            "distributions": split["distributions"],
        }
    source_files = {Path(info["path"]).resolve() for stage in (before, after) if stage for info in stage["source_artifacts"].values()}
    destinations = {output_dir / "report.html", output_dir / "comparison.json"}
    if destinations & (source_files | {args.input.resolve()}):
        raise ValueError("Comparison output would overwrite source artifacts")
    links = []
    for stage_name, directory in (("训练前", args.before_run_dir), ("训练后", args.after_run_dir)):
        if directory:
            for name in ("metrics.json", "predictions.jsonl", "run_metadata.json"):
                target = directory / name
                if target.exists():
                    links.append((stage_name + " " + name, quote(Path(os.path.relpath(target.resolve(), output_dir)).as_posix(), safe="/")))
    page = render_html(comparison, artifact_links=links)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (("comparison.json", json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n"), ("report.html", page)):
        destination = output_dir / name
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    for stage in (before, after):
        if stage:
            for artifact in stage["source_artifacts"].values():
                if _sha(Path(artifact["path"]).read_bytes()) != artifact["sha256"]:
                    raise ValueError(f"Source artifact changed during comparison generation: {artifact['path']}")
    if _sha(args.input.read_bytes()) != input_sha256:
        raise ValueError("Frozen input changed during comparison generation")
    if diagnostic_verification and _sha(diagnostic_path.read_bytes()) != diagnostic_verification["diagnostic_artifact_sha256"]:
        raise ValueError("Format diagnostics changed during comparison generation")
    print(json.dumps({"status": comparison["status"], "output_dir": str(output_dir),
                      "before_joint_three": before["metrics"]["joint_three"]["accuracy"],
                      "after_joint_three": after["metrics"]["joint_three"]["accuracy"] if after else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
