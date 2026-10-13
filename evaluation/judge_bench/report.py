"""Render a self-contained Chinese JudgeBench report from saved run artifacts.

Example:
    python -B -m evaluation.judge_bench.report --run-dir outputs/judge_bench_v1/runs/qwen3.5-9b \
        --manifest outputs/judge_bench_v1/split_manifest.json \
        --output outputs/judge_bench_v1/report.html

Pass --baseline-run-dir to compare a preserved first run with a retried run.

This module reads saved evidence and metrics only; it never calls a model.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT_DIR / "outputs/judge_bench_v1/split_manifest.json"
DEFAULT_OUTPUT = ROOT_DIR / "outputs/judge_bench_v1/report.html"
DOMAIN_NAMES = {
    "community_and_civic_life": "社区与公共生活",
    "education_and_learning": "教育与学习",
    "family_and_caregiving_life": "家庭与照护",
    "health_and_wellbeing": "健康与身心状态",
    "home_and_household_management": "居家与家务管理",
    "personal_finance_and_consumer_life": "个人财务与消费",
    "recreation_sports_and_hobbies": "休闲、运动与爱好",
    "social_and_cultural_life": "社交与文化生活",
    "travel_and_mobility": "旅行与出行",
    "work_and_professional_life": "工作与职业生活",
}
LABEL_NAMES = {
    "acceptable": "可接受", "unacceptable": "不可接受", "__failed__": "失败／缺失",
    "blockage": "阻碍", "dependency": "依赖", "opportunity": "机会", "risk": "风险",
    "transport_error": "传输失败", "api_error": "API 失败", "schema_error": "结构失败",
    "missing": "缺失", "ok": "有效输出",
}


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any, places: int = 2) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{value:,.{places}f}"


def _integer(value: Any) -> str:
    return _number(value, 0)


def _percent(value: Any) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    rounded = (Decimal(str(value)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{rounded:.2f}%"


def _duration(value: Any) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    seconds = round(value)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (f"{hours} 小时 " if hours else "") + f"{minutes} 分 {seconds} 秒"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    heading = "".join(f"<th scope=\"col\">{_e(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_e(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    if not rows:
        body = f"<tr><td colspan=\"{len(headers)}\">未记录</td></tr>"
    return f"<div class=\"table-wrap\"><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>"


def _card(label: str, value: str, note: str = "") -> str:
    return f"<article class=\"card\"><span>{_e(label)}</span><strong>{_e(value)}</strong><small>{_e(note)}</small></article>"


def _ci(intervals: dict, name: str, *, percent: bool = True) -> str:
    interval = intervals.get(name)
    if not isinstance(interval, list) or len(interval) != 2 or None in interval:
        return "95% CI：未记录"
    formatter = _percent if percent else _number
    return f"95% CI：{formatter(interval[0])} – {formatter(interval[1])}"


def _gpu_summary(sessions: list[dict]) -> dict:
    valid = [session for session in sessions if session.get("gpu_utilization_percent", {}).get("n", 0)]
    n = sum(session["gpu_utilization_percent"]["n"] for session in valid)
    return {
        "samples": n,
        "mean": sum(
            session["gpu_utilization_percent"]["mean"] * session["gpu_utilization_percent"]["n"]
            for session in valid
        ) / n if n else None,
        # Session percentiles cannot be merged into an exact pooled percentile.
        "p50": valid[0]["gpu_utilization_percent"]["p50"] if len(valid) == 1 else None,
        "max": max((session["gpu_utilization_percent"]["max"] for session in valid), default=None),
        "fraction_90": sum(
            session.get("fraction_samples_gpu_utilization_at_least_90", 0)
            * session["gpu_utilization_percent"]["n"]
            for session in valid
        ) / n if n else None,
        "session_count": len(valid),
    }


def render_report(
    metrics: dict, metadata: dict, manifest: dict, frozen_prompt: dict,
    *, artifact_links: list[tuple[str, str]] | None = None,
    baseline_metrics: dict | None = None, baseline_metadata: dict | None = None,
) -> str:
    """Return escaped standalone HTML, permitting verification entirely in memory."""
    config = metadata.get("config", {})
    model = config.get("model", "未知模型")
    total = metrics.get("total", config.get("input_rows"))
    verdict = metrics.get("verdict", {})
    bootstrap = metrics.get("verdict_confidence_intervals", {})
    intervals = bootstrap.get("intervals", {})
    performance = metrics.get("performance", metadata.get("performance", {}))
    final_finish_reasons = performance.get("final_finish_reason_counts", performance.get("finish_reason_counts", {}))
    retry_invocations = [entry for entry in (metadata.get("retry_invocations") or []) if entry.get("mode") == "retry_failed"]
    lineage = metadata.get("retry_lineage") or {}
    retry_targets = set(lineage.get("eligible_item_ids", []))
    retry_targets.update(item_id for entry in retry_invocations for item_id in entry.get("item_ids", []))
    gpu_sessions = metrics.get("gpu_sessions", [session.get("gpu", {}) for session in metadata.get("sessions", [])])
    gpu = _gpu_summary(gpu_sessions)
    test = manifest.get("distributions", {}).get("test", {})
    train = manifest.get("distributions", {}).get("train", {})
    overlaps = manifest.get("overlap", {})
    audit = manifest.get("duplicate_annotation_audit", {}).get("test", {})
    complete = metrics.get("run_complete", metadata.get("complete", False))
    status = "已完成" if complete else "尚未完成；缺失输出计入错误"
    cards = "".join([
        _card("判决准确率", _percent(verdict.get("accuracy")), _ci(intervals, "accuracy")),
        _card("判决 Macro F1", _percent(verdict.get("macro_f1")), _ci(intervals, "macro_f1")),
        _card("判决 Balanced Accuracy", _percent(verdict.get("balanced_accuracy")), _ci(intervals, "balanced_accuracy")),
        _card("Cohen’s κ", _number(verdict.get("cohen_kappa")), _ci(intervals, "cohen_kappa", percent=False)),
    ])
    diagnostic_rows = []
    for field, name in (
        ("timing", "Timing"), ("intensity", "Intensity（人工标签非空）"),
        ("action_intensity", "Action intensity（L0–L3）"),
        ("other_issue_including_null", "Other issue（含无其他问题）"),
    ):
        score = metrics.get(field, {})
        diagnostic_rows.append([
            name, _integer(score.get("n")), _percent(score.get("accuracy")),
            _percent(score.get("macro_f1")),
        ])
    consistency = metrics.get("diagnostic_consistency", {})
    strict = metrics.get("verdict_with_strict_consistency", {})
    consistency_cards = "".join([
        _card("诊断内部一致率", _percent(consistency.get("consistency_rate_checked")),
              f"{_integer(consistency.get('consistent'))} / {_integer(consistency.get('checked'))} 个已检查输出"),
        _card("判决正确且诊断一致", _percent(strict.get("accuracy")),
              f"严格联合通过率；分母 {_integer(strict.get('n', total))}"),
        _card("真实失败／缺失", _integer(metrics.get("failed_predictions")),
              "传输、API、JSON 结构失败及缺失输出"),
        _card("最终条目达到长度上限", _integer(final_finish_reasons.get("length", 0)),
              "最终一次结果的 finish_reason=length；不等于结构失败数"),
    ])
    dimensions = metrics.get("rejection_dimensions", {})
    consistency_note = (
        f"内部不一致 {_integer(consistency.get('inconsistent', 0))} 条；"
        f"未检查 {_integer(consistency.get('not_checked', 0))} 条。"
        "主判决分数照常计入结构有效的原始判决，辅助标签不做修正。"
        "严格联合通过率要求判决正确并通过内部一致性检查；不等于所有标签都与人工标签完全一致。"
        f"Rejection dimensions：集合准确率 {_percent(dimensions.get('exact_match_accuracy'))}，"
        f"Micro F1 {_percent(dimensions.get('micro_f1'))}。"
    )
    confusion = verdict.get("confusion", {})
    confusion_labels = confusion.get("rows_gold", [])
    confusion_columns = confusion.get("columns_prediction", [])
    confusion_rows = [
        [LABEL_NAMES.get(label, label), *values]
        for label, values in zip(confusion_labels, confusion.get("counts", []))
    ]
    confusion_table = _table(
        ["人工标签 ↓ / 模型预测 →", *[LABEL_NAMES.get(label, label) for label in confusion_columns]],
        confusion_rows,
    )
    gold_verdicts = test.get("gold", {}).get("verdict", {})
    majority_label, majority_count = max(gold_verdicts.items(), key=lambda pair: pair[1], default=("—", 0))
    majority_accuracy = majority_count / total if total else None
    majority_macro_f1 = (2 * majority_count / (total + majority_count) / len(gold_verdicts)
                         if total and gold_verdicts else None)
    valid_only = metrics.get("verdict_valid_only", {})
    valid_note = (
        f"本次全部 {_integer(total)} 条均产生有效输出，有效输出分数与主分数相同。"
        if valid_only.get("n") == total else
        f"仅统计 {_integer(valid_only.get('n'))} 条有效输出时，判决准确率为 {_percent(valid_only.get('accuracy'))}；"
        "该有效输出分数使用不同分母，不能替代主分数。"
    )
    baseline_note = (
        f"全部预测为多数类“{LABEL_NAMES.get(majority_label, majority_label)}”的准确率为 {_percent(majority_accuracy)}"
        f"（{_integer(majority_count)} / {_integer(total)}），Macro F1 为 {_percent(majority_macro_f1)}。"
        + valid_note
    )
    retry_comparison = ""
    if baseline_metrics is not None:
        baseline_metadata = baseline_metadata or {}
        old_performance = baseline_metrics.get("performance", baseline_metadata.get("performance", {}))
        comparison_rows = []
        for name, score, perf in (
            ("首轮（保留原目录）", baseline_metrics, old_performance),
            ("结构重试后（累计）", metrics, performance),
        ):
            comparison_rows.append([
                name, _percent(score.get("verdict", {}).get("accuracy")),
                _percent(score.get("verdict", {}).get("macro_f1")),
                f"{_integer(score.get('valid_predictions'))} / {_integer(score.get('failed_predictions'))}",
                _integer(perf.get("request_attempts")), _integer(perf.get("tokens", {}).get("total_tokens")),
                _duration(perf.get("active_wall_seconds")),
            ])

        def difference(field: str, *, token: bool = False) -> float | None:
            old = old_performance.get("tokens", {}).get(field) if token else old_performance.get(field)
            new = performance.get("tokens", {}).get(field) if token else performance.get(field)
            return new - old if isinstance(new, (int, float)) and isinstance(old, (int, float)) else None

        retry_cards = "".join([
            _card("重试目标条目", _integer(len(retry_targets)), "按各重试批次 item_ids 去重"),
            _card("新增 API 尝试", _integer(difference("request_attempts")), "含结构重试及传输自动重试"),
            _card("新增 token 成本", _integer(difference("total_tokens", token=True)),
                  f"输入 {_integer(difference('prompt_tokens', token=True))} / 输出 {_integer(difference('completion_tokens', token=True))}"),
            _card("重试新增耗时", _duration(difference("active_wall_seconds"))),
        ])
        policies = sorted({entry.get("max_structure_retries") for entry in retry_invocations if entry.get("max_structure_retries") is not None})
        policy_text = " / ".join(str(value) for value in policies) or "未记录"
        policy_rows = [[
            entry.get("invocation_id", index), _integer(entry.get("target_count", len(entry.get("item_ids", [])))),
            _integer(entry.get("max_structure_retries")), _integer(entry.get("max_transport_retries")),
            _integer(entry.get("request_attempts")), entry.get("seed_strategy", "未记录"),
        ] for index, entry in enumerate(retry_invocations, start=1)]
        retry_comparison = (
            "<section><h2>首轮与结构重试对照</h2>"
            + _table(["阶段", "判决准确率", "Macro F1", "有效 / 失败", "全部 API 尝试", "累计 tokens", "累计耗时"], comparison_rows)
            + f"<div class=\"grid\">{retry_cards}</div>"
            + "<p class=\"note\">原首轮结果单独保留，下方提供原目录及原始文件链接。只对结构失败条目重试；"
              "prompt、context、解码配置不变，使用不同的可复现采样 seed，取首个结构有效输出计分。"
              f"记录的 max_structure_retries={_e(policy_text)}，表示本批首个请求后允许的额外结构重试次数；"
              "历史首轮另行保留，传输自动重试预算另计。辅助诊断不一致继续单独记录，不修正原始标签。"
              "累计请求、tokens、耗时与 GPU 采样包含首轮和全部重试；未返回用量的请求失败无法计入 token 成本。</p>"
            + (_table(
                ["重试批次", "目标数", "额外结构重试上限", "额外传输重试上限", "实际 API 尝试", "Seed 策略"], policy_rows,
            ) if policy_rows else "")
            + "</section>"
        )
    class_rows = [
        [LABEL_NAMES.get(label, label), _integer(score.get("support")),
         _percent(score.get("precision")), _percent(score.get("recall")), _percent(score.get("f1"))]
        for label, score in verdict.get("per_class", {}).items()
    ]
    breakdown_sections = []
    for field, title in (
        ("source_model", "候选行动来源模型"), ("domain_id", "领域"), ("trigger_type", "Trigger"),
    ):
        rows = []
        for label, score in sorted(metrics.get("breakdown", {}).get(field, {}).items()):
            display = DOMAIN_NAMES.get(label, LABEL_NAMES.get(label, label))
            rows.append([
                display, _integer(score.get("n")), _percent(score.get("accuracy")),
                _percent(score.get("macro_f1")), _percent(score.get("balanced_accuracy")),
            ])
        breakdown_sections.append(f"<h3>{_e(title)}</h3>" + _table(
            ["分组", "条数", "判决准确率", "Macro F1", "Balanced Accuracy"], rows,
        ))
    gpu_note = (
        f"{gpu['samples']:,} 次 GPU 采样；均值按有效样本数合并。"
        + ("多个运行时段的 p50 分别列出，不合并分位数。" if gpu["session_count"] > 1 else "")
        if gpu["samples"] else "本次运行未记录 GPU 采样。"
    )
    if baseline_metrics is not None or retry_invocations:
        gpu_note += " 全部首轮与重试运行时段一起统计。"
    gpu_cards = "".join([
        _card("GPU 平均利用率", _number(gpu["mean"]) + ("%" if gpu["mean"] is not None else "")),
        _card("GPU p50 / 峰值", " / ".join(
            _number(value) + ("%" if value is not None else "") for value in (gpu["p50"], gpu["max"])
        )),
        _card("GPU ≥90% 的采样比例", _percent(gpu["fraction_90"])),
        _card("累计运行耗时", _duration(performance.get("active_wall_seconds", metadata.get("active_wall_seconds")))),
    ])
    gpu_rows = []
    for index, session in enumerate(gpu_sessions, start=1):
        util = session.get("gpu_utilization_percent", {})
        gpu_rows.append([
            index, _integer(util.get("n", 0)), _number(util.get("mean")),
            _number(util.get("p50")), _number(util.get("max")),
            _percent(session.get("fraction_samples_gpu_utilization_at_least_90")),
        ])
    gpu_table = _table(["运行时段", "有效采样", "平均 %", "p50 %", "峰值 %", "≥90% 比例"], gpu_rows) if gpu["session_count"] > 1 else ""
    tokens = performance.get("tokens", {})
    throughput_table = _table(["项目", "数值"], [
        ["全部尝试输入 / 输出 tokens", f"{_integer(tokens.get('prompt_tokens'))} / {_integer(tokens.get('completion_tokens'))}"],
        ["全部尝试总 tokens（成本）", _integer(tokens.get("total_tokens"))],
        ["输出 tokens/s（全部尝试）", _number(performance.get("completion_tokens_per_second"))],
        ["总 tokens/s（全部尝试）", _number(performance.get("total_tokens_per_second"))],
        ["API 尝试/s", _number(performance.get("requests_per_second"))],
        ["API 尝试次数（含全部失败及重试）", _integer(performance.get("request_attempts"))],
        ["全部尝试达到长度上限次数", _integer(performance.get("finish_reason_counts", {}).get("length", 0))],
        ["最终条目达到长度上限条数", _integer(final_finish_reasons.get("length", 0))],
        ["最终有效 / 失败条数", f"{_integer(metrics.get('valid_predictions'))} / {_integer(metrics.get('failed_predictions'))}"],
        ["API 尝试延迟 p50 / p95（秒）", " / ".join(_number(performance.get("latency_seconds", {}).get(key)) for key in ("p50", "p95"))],
        ["独立 reasoning token 计数", _integer(tokens.get("reasoning_tokens")) if tokens.get("reasoning_tokens") else "API 未提供独立计数或计数为 0"],
    ])
    split_rows = [
        ["测试集", _integer(test.get("items")), _integer(test.get("scenarios")), "80 / 80 / 200"],
        ["训练集", _integer(train.get("items")), _integer(train.get("scenarios")), "720 / 720 / 1800"],
    ]
    shared_instances = overlaps.get("instance_key", {}).get("shared_count")
    shared_scenarios = overlaps.get("scenario_key", {}).get("shared_count")
    protocol_note = (
        f"固定 seed={manifest.get('seed')}；40 个领域 × trigger 分层各留出 1 个完整 scenario。"
        "每个 scenario 的 20 条人工标注一起进入同一集合，选择不依赖人工标签或模型预测。"
        f"两集合共享 scenario {_integer(shared_scenarios)} 个，共享用户 instance {_integer(shared_instances)} 个；"
        "这是 scenario 留出，用户背景与部分可复用记录可能共享。"
    )
    quality_note = (
        f"测试集有 {_integer(audit.get('conflicting_gold_groups', 0))} 组完全相同输入的人工标签存在分歧，"
        "标签原样保留。对相同输入给出一致判决时，"
        f"准确率上限为 {_percent(audit.get('maximum_verdict_accuracy_for_consistent_predictions'))}。"
    )
    config_rows = [
        ["模型", model], ["Prompt 版本", config.get("prompt_version", frozen_prompt.get("prompt_version", "—"))],
        ["Prompt SHA-256", config.get("prompt_sha256", frozen_prompt.get("prompt_sha256", "—"))],
        ["测试集 SHA-256", config.get("input_sha256", metrics.get("input_sha256", "—"))],
        ["并发 / max_tokens", f"{config.get('concurrency', '—')} / {config.get('max_tokens', '—')}"],
        ["temperature / top_p", f"{config.get('temperature', '—')} / {config.get('top_p', '—')}"],
        ["presence_penalty / top_k", f"{config.get('presence_penalty', '—')} / {config.get('top_k') if config.get('top_k') is not None else '未指定'}"],
        ["thinking / structured_output", f"{config.get('thinking', '—')} / {config.get('structured_output', '—')}"],
        ["采样 seed", config.get("seed", "—")],
        ["网络传输", config.get("transport", "HTTP(S)")],
    ]
    status_rows = [[LABEL_NAMES.get(label, label), count] for label, count in sorted(metrics.get("status_counts", {}).items())]
    consistency_errors = [[label, count] for label, count in sorted(consistency.get("error_counts", {}).items())]
    consistency_details = "<details><summary>内部一致性检查详情</summary>" + _table(["问题", "条数"], consistency_errors) + "</details>" if consistency_errors else ""
    artifact_html = "".join(
        f"<a href=\"{_e(href)}\">{_e(label)}</a>" for label, href in (artifact_links or [])
    )
    title = f"{model} · JudgeBench"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title><style>
:root{{--ink:#173044;--muted:#5f7180;--line:#dce5ec;--paper:#f3f6f8;--accent:#14665d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1140px;margin:0 auto;padding:40px 24px 64px}}header{{margin-bottom:26px}}h1{{font-size:32px;line-height:1.25;margin:10px 0}}h2{{font-size:21px;margin:0 0 18px}}h3{{font-size:16px;margin:24px 0 10px}}
p{{margin:12px 0}}.eyebrow{{color:var(--accent);font-weight:700;letter-spacing:.04em}}.muted,small{{color:var(--muted)}}.status{{display:inline-block;border:1px solid var(--line);border-radius:16px;padding:3px 12px;font-size:13px;background:white}}
section{{background:white;border:1px solid var(--line);border-radius:14px;padding:24px;margin:18px 0}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}}.card{{padding:17px;background:#f6f9fa;border:1px solid var(--line);border-radius:10px}}
.card span,.card strong,.card small{{display:block}}.card span{{font-size:13px;color:var(--muted)}}.card strong{{font-size:25px;margin:5px 0;line-height:1.35;font-variant-numeric:tabular-nums}}.card small{{font-size:12px;line-height:1.55}}
.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}}th{{background:#f5f8fa;color:var(--muted);font-size:13px}}td:not(:first-child){{min-width:86px}}
.note{{font-size:13px;color:var(--muted)}}.links{{display:flex;flex-wrap:wrap;gap:12px 22px}}a{{color:var(--accent);text-decoration:underline;text-underline-offset:3px}}details{{margin-top:16px}}summary{{cursor:pointer;color:var(--muted)}}footer{{font-size:13px;color:var(--muted);margin-top:20px}}
@media(max-width:850px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:480px){{main{{padding:24px 12px}}section{{padding:18px 14px}}h1{{font-size:26px}}.grid{{grid-template-columns:1fr}}}}
@media print{{body{{background:white}}main{{max-width:none;padding:0}}section{{break-inside:avoid}}details{{display:none}}}}
</style></head><body><main>
<header><div class="eyebrow">JUDGEBENCH · {_e(manifest.get('benchmark_id', ''))}</div><h1>{_e(title)}</h1>
<p class="muted">对人工标注的 proactive action 进行独立判决 · {_e(_integer(total))} 条测试样本</p><span class="status">{_e(status)}</span></header>
<section><h2>主判决表现</h2><div class="grid">{cards}</div>
<p class="note">准确率分母包含全部测试条目；结构失败和缺失输出计入错误。Macro F1 对两种判决等权。95% CI 按完整 scenario 做 cluster bootstrap：{_e(bootstrap.get('scenario_groups', '—'))} 组，{_e(bootstrap.get('repeats', '—'))} 次重采样。</p>
<p>{_e(baseline_note)}</p>{confusion_table}
{_table(['人工判决', '条数', 'Precision', 'Recall', 'F1'], class_rows)}</section>
{retry_comparison}
<section><h2>辅助诊断与输出质量</h2>{_table(['指标', '计分条数', '准确率', 'Macro F1'], diagnostic_rows)}
<p class="note">Intensity 仅对人工标签非空的条目计分；被拒绝的 L0 使用 Inaction / intensity=null。</p>
<div class="grid">{consistency_cards}</div><p class="note">{_e(consistency_note)}</p>
{_table(['输出状态', '条数'], status_rows)}{consistency_details}</section>
<section><h2>分组判决表现</h2><p class="note">来源模型表示被评价行动的生成模型；本报告的 judge 为 {_e(model)}。各组使用自身全部条目作分母。</p>{''.join(breakdown_sections)}</section>
<section><h2>GPU 与吞吐</h2><div class="grid">{gpu_cards}</div><p class="note">{_e(gpu_note)}</p>{gpu_table}{throughput_table}
<p class="note">token 成本与吞吐统计全部请求尝试，包含首轮失败、结构重试和传输自动重试返回的用量；耗时与 GPU 统计使用全部运行时段。输出 token 可能包含 thinking，独立 reasoning 计数以 API 报告为准。全部尝试达到长度上限的次数与最终条目达到长度上限的条数分别列出，均不等于结构失败条数。</p></section>
<section><h2>固定数据划分与评测协议</h2>{_table(['集合', '条数', 'Scenario 数', '每模型 / 每领域 / 每 trigger'], split_rows)}
<p>{_e(protocol_note)}</p><p>输入与人工页面对齐：background 包含完整的 instance_description、personal_description、scenario_description；另提供 checkpoint prefix、按轮次排序的 retrieval command/result 和 candidate action。保持全部已有证据，不截断 context。人工标签、标注评论和来源信息保存在独立字段中，不进入模型输入。</p>
<p class="note">{_e(quality_note)}</p></section>
<section><h2>可复现配置与原始文件</h2>{_table(['配置', '值'], config_rows)}<p class="links">{artifact_html}</p></section>
<footer>报告从保存的 metrics、运行配置与数据划分生成；不执行推理，也不修改 prompt、标签或数据选择。</footer>
</main></body></html>"""


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _relative_link(output: Path, target: Path) -> str:
    return quote(Path(os.path.relpath(target.resolve(), output.parent.resolve())).as_posix(), safe="/")


def _attempt_accounting(path: Path) -> dict:
    """Read all retained attempts without changing an older run's accounting."""
    tokens = {key: 0 for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")}
    finish_reasons: Counter[str] = Counter()
    attempts = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for attempt in record.get("attempts", []):
                attempts += 1
                response = attempt.get("response") or {}
                usage = response.get("usage") or {}
                tokens["prompt_tokens"] += usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
                tokens["completion_tokens"] += usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
                tokens["total_tokens"] += usage.get("total_tokens", 0) or 0
                detail = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
                tokens["reasoning_tokens"] += detail.get("reasoning_tokens", 0) or 0
                choices = response.get("choices") or []
                if choices:
                    finish_reasons[str(choices[0].get("finish_reason"))] += 1
    return {"tokens": tokens, "request_attempts": attempts, "finish_reason_counts": dict(finish_reasons)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, help="Preserved first-run artifacts for comparison")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    metrics = _load(args.run_dir / "metrics.json")
    metadata = _load(args.run_dir / "run_metadata.json")
    manifest = _load(args.manifest)
    frozen_prompt = _load(args.run_dir / "frozen_prompt.json")
    if metadata.get("config", {}).get("prompt_sha256") != frozen_prompt.get("prompt_sha256"):
        raise ValueError("Saved run and frozen prompt hashes disagree")
    baseline_metrics = baseline_metadata = None
    if args.baseline_run_dir:
        baseline_metrics = _load(args.baseline_run_dir / "metrics.json")
        baseline_metadata = _load(args.baseline_run_dir / "run_metadata.json")
        for field in (
            "input_sha256", "prompt_sha256", "schema_sha256", "model", "max_tokens",
            "temperature", "top_p", "presence_penalty", "top_k", "thinking", "structured_output",
        ):
            if metadata.get("config", {}).get(field) != baseline_metadata.get("config", {}).get(field):
                raise ValueError(f"Baseline and current run differ in {field}")
        # Older metadata counts only final outputs. Recompute comparison costs
        # from retained raw attempts in memory, preserving the original directory.
        baseline_performance = dict(baseline_metrics.get("performance", baseline_metadata.get("performance", {})))
        raw_path = args.baseline_run_dir / "raw_responses.jsonl"
        if raw_path.exists():
            baseline_performance.setdefault("final_finish_reason_counts", baseline_performance.get("finish_reason_counts", {}))
            baseline_performance.update(_attempt_accounting(raw_path))
        baseline_metrics = {**baseline_metrics, "performance": baseline_performance}
    output = args.output.resolve()
    targets = [
        ("测试集（800）", args.manifest.parent / "test.jsonl"),
        ("训练集（7200）", args.manifest.parent / "train.jsonl"),
        ("数据划分 manifest", args.manifest),
        ("完整 metrics", args.run_dir / "metrics.json"),
        ("原始 predictions", args.run_dir / "predictions.jsonl"),
        ("原始 API responses", args.run_dir / "raw_responses.jsonl"),
        ("运行 metadata", args.run_dir / "run_metadata.json"),
        ("本次冻结 prompt", args.run_dir / "frozen_prompt.json"),
        ("Prompt 源码", ROOT_DIR / "evaluation/judge_bench/prompt.py"),
    ]
    if args.baseline_run_dir:
        targets.extend([
            ("首轮原始目录", args.baseline_run_dir),
            ("首轮 metrics", args.baseline_run_dir / "metrics.json"),
            ("首轮 predictions", args.baseline_run_dir / "predictions.jsonl"),
            ("首轮 API responses", args.baseline_run_dir / "raw_responses.jsonl"),
            ("首轮 metadata", args.baseline_run_dir / "run_metadata.json"),
        ])
    links = [(label, _relative_link(output, target)) for label, target in targets if target.exists()]
    content = render_report(
        metrics, metadata, manifest, frozen_prompt, artifact_links=links,
        baseline_metrics=baseline_metrics, baseline_metadata=baseline_metadata,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(output), "model": metadata.get("config", {}).get("model"), "total": metrics.get("total")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
