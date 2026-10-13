"""Compare thinking on/off for identical trained weights and frozen judge inputs.

This is an inference-mode ablation, not a training improvement comparison.
All source artifacts are read-only. Failures remain errors in full-set metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from urllib.parse import quote

from .compare import CORE, _accuracy_delta_pp, _load_run, _read_jsonl, check_match, paired_valid_analysis
from .report import _percent, _table
from .runner import _atomic_json


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_modes(before: dict, after: dict, before_predictions: list[dict],
                 after_predictions: list[dict], before_metadata: dict, after_metadata: dict) -> dict:
    match = check_match(before, after, before_predictions, after_predictions)
    if match["mismatches"] != ["thinking"]:
        raise ValueError("Mode ablation must differ only in thinking: " + str(match["mismatches"]))
    left, right = before["config"], after["config"]
    if left.get("thinking") != "true" or right.get("thinking") != "false":
        raise ValueError("Expected thinking=true before and thinking=false after")
    other_changes = [key for key in set(left) | set(right)
                     if key != "thinking" and left.get(key) != right.get(key)]
    if other_changes:
        raise ValueError("Unexpected inference changes: " + str(other_changes))
    server = before_metadata.get("served_configuration")
    if not server or server != after_metadata.get("served_configuration"):
        raise ValueError("Thinking modes must use the exact same trained serving manifest")
    if not server.get("adaptation", {}).get("adapter_sha256"):
        raise ValueError("Trained weights must have adapter provenance")
    return {
        "status": "verified_thinking_only_ablation", "same_trained_weights": True,
        "same_input_prompt_messages_and_other_settings": True,
        "only_changed_field": {"thinking": {"before": "true", "after": "false"}},
        "model_path": server["model_path"], "adapter_sha256": server["adaptation"]["adapter_sha256"],
        "strict_identical_inference_match": False,
        "interpretation": "Same trained checkpoint under different inference modes; no new training or weights change.",
        "retry_note": match["retry_note"],
    }


def render_report(result: dict, links: list[tuple[str, str]]) -> str:
    before, after = result["thinking_on"], result["thinking_off"]
    names = {"verdict": "接受判断", "timing": "Timing", "intensity": "Intensity（非空）",
             "joint_three": "接受、Timing、Intensity 三项同时正确"}
    rows = []
    for key, _ in CORE:
        a, b = before["metrics"][key], after["metrics"][key]
        delta = result["accuracy_delta_percentage_points"][key]
        delta_text = f"{Decimal(str(delta)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):+.2f}"
        rows.append([names[key], f"{a['correct']}/{a['n']} · {_percent(a['accuracy'])}",
                     f"{b['correct']}/{b['n']} · {_percent(b['accuracy'])}", delta_text])
    failures = _table(["输出状态", "Thinking 开启", "Thinking 关闭"], [
        ["有效输出", before["metrics"]["valid_predictions"], after["metrics"]["valid_predictions"]],
        ["失败输出", before["metrics"]["failed_predictions"], after["metrics"]["failed_predictions"]],
    ])
    recalls = []
    for field in ("verdict", "timing", "intensity"):
        for label, values in before["metrics"][field]["per_class"].items():
            new = after["metrics"][field]["per_class"][label]
            recalls.append([field, label, values["support"], _percent(values["recall"]),
                            _percent(new["recall"]), values["predicted"], new["predicted"]])
    subsets = []
    for key, label in (("timing_non_timely", "非 Timely"),
                       ("intensity_non_appropriate", "非 Appropriate（非空）"),
                       ("intensity_null_cases", "Intensity 为 null")):
        a, b = before["metrics"][key], after["metrics"][key]
        subsets.append([label, a["n"], _percent(a["accuracy"]), _percent(b["accuracy"])])
    paired = result["paired_valid_subset"]
    paired_rows = [[names[key], paired["before"][key]["n"],
                    _percent(paired["before"][key]["accuracy"]),
                    _percent(paired["after"][key]["accuracy"])] for key, _ in CORE]
    bias_rows = []
    for key, label in (("timing_correct_from_Timely", "Timing 正确结果中 Timely 占比"),
                       ("intensity_correct_from_Appropriate", "Intensity 正确结果中 Appropriate 占比")):
        a, b = before["correct_sources"][key], after["correct_sources"][key]
        bias_rows.append([label, f"{a['correct']}/{a['total_correct']} · {_percent(a['fraction'])}",
                          f"{b['correct']}/{b['total_correct']} · {_percent(b['fraction'])}"])
    artifact_links = " · ".join(f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>'
                                for label, url in links)
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>JudgeBench：已训练 Qwen3.5-9B 的 thinking 对照</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;background:#f5f7fa;color:#18253a;margin:0}}main{{max-width:1150px;margin:auto;padding:32px 20px}}section{{background:white;border:1px solid #dce3ec;border-radius:12px;padding:22px;margin:22px 0}}h1{{font-size:28px}}h2{{font-size:21px}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:left;border-bottom:1px solid #e4e9f0;padding:10px;white-space:nowrap}}th{{background:#f0f4f9}}a{{color:#245db9}}.note{{color:#52647b}}</style>
<main><h1>已训练 Qwen3.5-9B：Thinking 开启与关闭</h1>
<p>同一最终 LoRA 合并模型、同一800条测试集、同一 prompt 和证据。此次仅将 thinking 从 true 改为 false；并发64、输出上限16384、温度1、top_p 0.95、top_k 20、presence_penalty 1.5、seed 20261010 及原结构恢复协议保持一致。</p>
<p class="note">这是推理模式对照，模型权重没有变化。不同模式和结构恢复会产生不同的随机输出，不解释为新增训练的提升。</p>
<section><h2>全测试集准确率</h2>{_table(['指标','Thinking 开启','Thinking 关闭','变化（百分点）'],rows)}
<p class="note">失败输出计入错误。Intensity 使用772条人工非空标签；其余主指标分母800。三项同时正确要求 null 与人工标签一致。</p>{failures}</section>
<section><h2>少数类别与多数类偏向</h2>{_table(['子集','条数','Thinking 开启','Thinking 关闭'],subsets)}
{_table(['字段','人工类别','人工条数','开启召回率','关闭召回率','开启预测数','关闭预测数'],recalls)}
{_table(['正确结果来源','Thinking 开启','Thinking 关闭'],bias_rows)}</section>
<section><h2>两种模式都成功输出的同一批条目</h2>{_table(['指标','分母','Thinking 开启','Thinking 关闭'],paired_rows)}
<p class="note">共同有效条目 {paired['items']}/800。该子集受输出成功与否筛选影响，用于区分格式恢复和标签判断，不能代替上面的全测试集分数。</p></section>
<section><h2>原始文件与复现检查</h2><p>全部800条 item IDs、context/message hashes、冻结 prompt/schema 和模型来源已核验。除 thinking 外的推理设置相同。</p><p>{artifact_links}</p></section></main></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--thinking-on-run", type=Path, required=True)
    parser.add_argument("--thinking-off-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = _read_jsonl(args.input)
    if len(rows) != 800 or len({row["item_id"] for row in rows}) != 800:
        raise ValueError("Requires the original 800 unique JudgeBench items")
    input_sha = _sha(args.input)
    before, before_predictions = _load_run(args.thinking_on_run, rows, input_sha)
    after, after_predictions = _load_run(args.thinking_off_run, rows, input_sha)
    left_meta = json.loads((args.thinking_on_run / "run_metadata.json").read_text())
    right_meta = json.loads((args.thinking_off_run / "run_metadata.json").read_text())
    check = verify_modes(before, after, before_predictions, after_predictions, left_meta, right_meta)
    for run, analysis in ((args.thinking_on_run, before), (args.thinking_off_run, after)):
        path = run / "raw_responses.jsonl"
        analysis["source_artifacts"][path.name] = {"path": str(path.resolve()), "sha256": _sha(path)}
    result = {"format_version": "judge-bench-thinking-ablation-v1",
              "generated_utc": datetime.now(timezone.utc).isoformat(), "verification": check,
              "input": {"path": str(args.input.resolve()), "sha256": input_sha, "items": 800},
              "thinking_on": before, "thinking_off": after,
              "accuracy_delta_percentage_points": {key: _accuracy_delta_pp(before["metrics"][key], after["metrics"][key])
                                                   for key, _ in CORE},
              "paired_valid_subset": paired_valid_analysis(rows, before_predictions, after_predictions)}
    out = args.output_dir.resolve()
    sources = {Path(a["path"]).resolve() for analysis in (before, after) for a in analysis["source_artifacts"].values()}
    if {out / "comparison.json", out / "report.html"} & (sources | {args.input.resolve()}):
        raise ValueError("Report would overwrite source data")
    links = [("机器可读对照", "comparison.json")]
    for label, run in (("开启", args.thinking_on_run), ("关闭", args.thinking_off_run)):
        for name in ("metrics.json", "predictions.jsonl", "raw_responses.jsonl", "run_metadata.json"):
            url = quote(Path(os.path.relpath((run / name).resolve(), out)).as_posix(), safe="/")
            links.append((f"{label} {name}", url))
    page = render_report(result, links)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "comparison.json", result)
    tmp = out / "report.html.tmp"
    try:
        tmp.write_text(page, encoding="utf-8")
        tmp.replace(out / "report.html")
    finally:
        tmp.unlink(missing_ok=True)
    for analysis in (before, after):
        for artifact in analysis["source_artifacts"].values():
            if _sha(Path(artifact["path"])) != artifact["sha256"]:
                raise ValueError("Source artifact changed while generating the report")
    if _sha(args.input) != input_sha:
        raise ValueError("Frozen test input changed while generating the report")
    print(json.dumps({"status": check["status"], "output_dir": str(out),
                      "thinking_off_metrics": {key: after["metrics"][key]["accuracy"] for key, _ in CORE},
                      "failed_predictions": after["metrics"]["failed_predictions"]}))


if __name__ == "__main__":
    main()
