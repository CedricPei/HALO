"""Report same-scenario GPT relay judgments without claiming matched zero-shot conditions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from . import compare, metrics, runner


def create_report(test: Path, output: Path) -> dict:
    rows = runner._read_jsonl(test)
    predictions = runner._read_jsonl(output / "predictions.jsonl")
    saved = json.loads((output / "metrics.json").read_text())
    metadata = json.loads((output / "run_metadata.json").read_text())
    examples = json.loads((output / "example_manifest.json").read_text())
    distinct_counts = examples["distinct_checkpoint_counts"]
    analysis = compare.analyze(rows, predictions)
    by_id = {record["item_id"]: record for record in predictions}
    valid_rows = [row for row in rows if by_id.get(row["item_id"], {}).get("status") == "ok"]
    valid = metrics.compute_metrics(valid_rows, predictions, bootstrap_repeats=0)
    raw = runner._read_jsonl(output / "raw_responses.jsonl")
    attempts = [attempt for record in raw for attempt in record["attempts"]]
    returned_models = Counter(attempt.get("returned_model", "unknown") for attempt in attempts if "response" in attempt)
    summary = {"condition": "gpt-6-sol; reasoning_effort=none; same-scenario four training demonstrations",
               "target_count": len(rows), "valid_count": len(valid_rows), "analysis": analysis,
               "valid_only_metrics": {field: valid[field] for field, _ in compare.CORE},
               "example_distinct_checkpoint_counts": examples["distinct_checkpoint_counts"],
               "returned_models": dict(returned_models),
               "comparability": "Same fixed v3 targets, but labeled few-shot demonstrations differ from Qwen zero-shot inputs; not a matched model-only comparison"}
    runner._atomic_json(output / "analysis.json", summary)
    main = compare._table(["指标", "正确 / 分母", "准确率", "仅有效输出的准确率", "多数类参考"], [
        [name, f"{saved[field]['correct']} / {saved[field]['n']}", compare._pct(saved[field]["accuracy"]),
         compare._pct(valid[field]["accuracy"]),
         compare._pct(saved["baselines"]["always_majority"][field]["accuracy"])]
        for field, name in compare.CORE])
    per_class = []
    for field in ("verdict", "timing", "intensity", "intensity_null_cases"):
        for label, score in saved[field]["per_class"].items():
            per_class.append([field, label, score["support"], score["predicted"],
                              compare._pct(score["recall"]), compare._pct(score["precision"]), compare._pct(score["f1"])])
    subsets = compare._table(["人工标签子集", "正确 / 分母", "准确率", "Balanced accuracy"], [
        [name, f"{saved[field]['correct']} / {saved[field]['n']}", compare._pct(saved[field]["accuracy"]),
         compare._pct(saved[field]["balanced_accuracy"])]
        for field, name in (("timing_non_timely", "非 Timely"),
                            ("intensity_non_appropriate", "Excessive / Insufficient"),
                            ("intensity_null_cases", "Intensity=null"))])
    source_rows = [[key, f"{value['correct']} / {value['total_correct']}", compare._pct(value["fraction"])]
                   for key, value in analysis["correct_sources"].items()]
    bias_rows = []
    for field in ("verdict", "timing", "intensity"):
        bias = saved["prediction_bias"][field]
        for label, count in bias["gold_counts"].items():
            bias_rows.append([field, label, count, bias["prediction_counts"].get(label, 0),
                              compare._pct(bias["gold_fractions"].get(label)),
                              compare._pct(bias["prediction_fractions_all_rows"].get(label, 0))])
    first = json.loads((output / "first_pass_metrics.json").read_text())
    failures = compare._table(["阶段", "有效", "JSON / schema 失败", "API 失败", "传输失败"], [
        [name, scores["valid_predictions"], scores["status_counts"].get("schema_error", 0),
         scores["status_counts"].get("api_error", 0), scores["status_counts"].get("transport_error", 0)]
        for name, scores in (("首轮", first), ("失败恢复后", saved))])
    perf = saved["performance"]
    config = metadata["config"]
    protocol = compare._table(["设置", "值"], [
        ["中转 / 精确模型", f"{config['endpoint']} / {config['model']}"],
        ["返回的模型标识", json.dumps(dict(returned_models), ensure_ascii=False)],
        ["推理 / 输出上限 / 并发", f"{config['reasoning_effort']} / {config['max_completion_tokens']} / {config['concurrency']}"],
        ["temperature / top_p / presence_penalty / seed", f"{config['temperature']} / {config['top_p']} / {config['presence_penalty']} / {config['seed']}"],
        ["输出", "原七字段严格 JSON schema；内部不一致单独统计，不修正、不因此重试"],
        ["结构恢复", "首轮结构错误不即时重试；仅失败 ID 进行一轮恢复，额外结构重试最多2次；每阶段传输重试最多2次"],
        ["实际 HTTP 尝试", perf["request_attempts"]],
        ["总输入 / 输出 tokens", f"{perf['tokens']['prompt_tokens']} / {perf['tokens']['completion_tokens']}"],
        ["独立 reasoning 用量详情", f"{perf['reasoning_usage_details_reported_attempts']} 次请求有报告；其余未知"],
        ["耗时（秒）", round(perf["active_wall_seconds"], 2)],
        ["目标数据 SHA-256", config["input_sha256"]], ["示例训练集 SHA-256", config["train_sha256"]],
        ["Prompt SHA-256", config["prompt_sha256"]], ["消息文件 SHA-256", config["prepared_messages_sha256"]],
    ])
    classes = compare._table(["字段", "标签", "人工数", "预测数", "召回率", "Precision", "F1"], per_class)
    confusions = "".join(f"<h3>{field}</h3>{compare._confusion(saved[field])}" for field in ("verdict", "timing", "intensity"))
    links = " · ".join(f'<a href="{name}">{label}</a>' for name, label in (
        ("analysis.json", "完整分析"), ("metrics.json", "指标"), ("predictions.jsonl", "预测"),
        ("raw_responses.jsonl", "全部原始响应"), ("prepared_messages.jsonl", "冻结请求消息"),
        ("example_manifest.json", "逐条示例来源"), ("frozen_prompt.json", "Prompt 与 schema"),
        ("run_metadata.json", "运行配置"), ("evaluation_protocol.json", "恢复协议")))
    content = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPT-6-sol · JudgeBench v3 · 四示例评测</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f6;color:#172d36;font:15px/1.7 system-ui,-apple-system,"PingFang SC",sans-serif}}main{{max-width:1200px;margin:auto;padding:28px 22px}}section{{background:white;border:1px solid #dbe3e6;border-radius:12px;margin:18px 0;padding:22px}}h1{{font-size:30px;line-height:1.3}}h2{{font-size:21px}}h3{{font-size:17px}}p{{overflow-wrap:anywhere}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #e3eaec;vertical-align:top;overflow-wrap:anywhere}}th{{background:#f4f7f8}}a{{color:#146f80}}.note{{color:#516873}}@media(max-width:600px){{main{{padding:18px 12px}}section{{padding:15px}}h1{{font-size:24px}}th,td{{min-width:80px}}}}
</style></head><body><main><h1>GPT-6-sol：新800条 JudgeBench</h1>
<p>Tiantoken HTTPS · reasoning_effort=none · 每条4个人工示例 · {len(valid_rows)} / {len(rows)} 有效输出</p>
<section><h2>评测条件</h2><p>训练集7200条和测试集800条都覆盖400个 scenario，每个 scenario 为18条训练、2条测试。本结果衡量固定 benchmark 内的专用 judge，不能解释为未见 scenario 的泛化表现。</p>
<p>示例只来自7200条训练集，选择过程不读取目标标签，也不按示例标签挑选。每条优先选4个其他 checkpoint；{distinct_counts.get('4', 0)}条提供4个不同 checkpoint，另外{distinct_counts.get('3', 0)}条提供3个其他 checkpoint 的4条不同标注。全部排除目标 checkpoint 的所有模型结果。background 只提供一次。示例的未来事件不作为目标已观测证据。</p>
<p>GPT 使用同 scenario 的人工示例，Qwen v3 的基座/训练后评测使用无示例输入。两者测试目标相同，但提示信息不同，因此不能将差值直接解释为纯模型优劣或训练提升。</p></section>
<section><h2>四项准确率</h2>{main}<p class="note">主分母包含全部800条，结构失败计错。Intensity 主分母为人工标签非空的769条；31条 null 单列。Joint3 要求同一条的接受判断、timing、intensity 全部正确，null 也必须匹配。多数类参考只根据人工分布计算。</p></section>
<section><h2>少数标签与正确来源</h2>{subsets}{compare._table(['正确来源','正确数 / 所有正确数','占比'], source_rows)}{classes}</section>
<section><h2>预测分布与多数类偏向</h2>{compare._table(['字段','类别','人工数','预测数','人工占比','预测占比'], bias_rows)}{confusions}</section>
<section><h2>输出失败与恢复</h2>{failures}<p>诊断内部不一致 {saved['diagnostic_consistency']['inconsistent']} 条，保留原判断计分。所有请求失败、结构错误和重试的原始响应与用量均保留。</p></section>
<section><h2>冻结配置与产物</h2>{protocol}<p class="note">中转未报告独立 reasoning token 数时记为未知。请求显式关闭 reasoning，不通过未报告的用量推断隐藏 token 为零。</p><p>{links}</p></section>
</main></body></html>'''
    (output / "report.html").write_text(content, encoding="utf-8")
    return summary
