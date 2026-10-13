"""Compare matched base/LoRA predictions on a checkpoint-selection validation set."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
from .compare import CORE, _accuracy_delta_pp, _load_run, check_match, paired_valid_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--base-run-dir', type=Path, required=True)
    parser.add_argument('--lora-run-dir', type=Path, required=True)
    parser.add_argument('--training-metadata', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = args.input.read_bytes()
    rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    if len(rows) != 400 or len({r['item_id'] for r in rows}) != 400 or any(r['split'] != 'validation' for r in rows):
        raise ValueError('Requires the frozen 400 unique validation items')
    digest = hashlib.sha256(data).hexdigest()
    before, bp = _load_run(args.base_run_dir, rows, digest)
    after, ap = _load_run(args.lora_run_dir, rows, digest)
    match = check_match(before, after, bp, ap)
    if match['matched'] is not True:
        raise ValueError('Base/LoRA inference settings or evidence differ: '+str(match['mismatches']))
    training = json.loads(args.training_metadata.read_text())
    if training.get('status') != 'complete' or training['data']['fit']['items'] != 7600 or training['data']['validation']['items'] != 400:
        raise ValueError('Training lineage differs from the requested 7600/400 run')
    result = {'format_version':'judge-bench-validation-comparison-v1',
        'generated_utc':datetime.now(timezone.utc).isoformat(), 'status':match['status'], 'match_check':match,
        'input':{'path':str(args.input.resolve()),'items':400,'sha256':digest},
        'before':before,'after':after,
        'accuracy_delta_percentage_points':{key:_accuracy_delta_pp(before['metrics'][key],after['metrics'][key]) for key,_ in CORE},
        'paired_valid_subset':paired_valid_analysis(rows,bp,ap),
        'checkpoint_selection':{'best_checkpoint':training['best_internal_validation_checkpoint'],
            'training_items':7600,'validation_items':400,'epochs':training['epochs_completed']},
        'interpretation':'These 400 rows were used to choose the LoRA checkpoint; scores are a validation-set comparison, not independent test generalization.',
        'scoring':'All 400 rows scored; missing or structurally invalid predictions count as incorrect. Intensity excludes null human labels; Joint3 includes null. Semantic mistakes are never retried.'}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    def e(value):return html.escape(str(value))
    def pct(value):return '—' if value is None else f'{value*100:.2f}%'
    def score(value):return f"{value['correct']}/{value['n']} · {pct(value['accuracy'])}"
    def table(headers,body):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+e(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+e(x)+'</td>' for x in row)+'</tr>' for row in body)+'</tbody></table></div>'
    core = [[title,score(before['metrics'][key]),score(after['metrics'][key]),f"{result['accuracy_delta_percentage_points'][key]:+.2f}"] for key,title in CORE]
    balanced=[];classes=[];subsets=[];quality=[]
    for name,run in [('Base',before),('LoRA',after)]:
        metrics=run['metrics']
        quality.append([name,metrics['valid_predictions'],metrics['failed_predictions'],score(metrics['joint_six']),score(metrics['verdict_with_strict_consistency'])])
        for key,title in [('verdict','可接受性'),('timing','时机'),('intensity','强度（人工非空）')]:
            m=metrics[key];balanced.append([name,title,score(m),pct(m['macro_f1']),pct(m['balanced_accuracy'])])
            for label,v in m['per_class'].items():
                classes.append([name,title,label,v['support'],v['predicted'],pct(v['precision']),pct(v['recall']),pct(v['f1'])])
        for key,title in [('timing_non_timely','非 Timely'),('intensity_non_appropriate','Excessive / Insufficient'),('intensity_null_cases','Intensity=null'),('intensity_including_null','Intensity，含 null')]:
            subsets.append([name,title,score(metrics[key]),pct(metrics[key]['macro_f1']),pct(metrics[key]['balanced_accuracy'])])
    links=[]
    for name,folder in [('Base',args.base_run_dir),('LoRA',args.lora_run_dir)]:
        for filename in ['predictions.jsonl','raw_responses.jsonl','metrics.json','run_metadata.json']:
            target=os.path.relpath(folder/filename,args.output_dir)
            links.append('<a href="'+e(target)+'">'+e(name+' · '+filename)+'</a>')
    model_title = Path(training['config']['model_path']).name
    report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen3.5-9B · 400条验证集对照</title><style>body{margin:0;background:#f4f6f8;color:#173044;font:15px/1.65 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1150px;margin:auto;padding:28px 20px}section{background:white;border:1px solid #dce5ec;border-radius:12px;padding:22px;margin:18px 0}h1{font-size:28px}h2{font-size:20px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:10px;border-bottom:1px solid #dce5ec;text-align:left;white-space:nowrap}th{background:#f5f8fa}.links{display:flex;gap:16px;flex-wrap:wrap}a{color:#14665d}code{overflow-wrap:anywhere}@media(max-width:600px){main{padding:15px 10px}section{padding:16px 12px}}</style><main><h1>Qwen3.5-9B：400 条验证集对照</h1>'''
    report=report.replace('Qwen3.5-9B',e(model_title))
    report+='<p>7600 条训练，400 条验证；原始 Base 与 LoRA 使用同一 400 条、同一完整证据、同一 prompt 和相同推理设置。匹配核验通过。</p><p>这 400 条参与了 checkpoint 选择，因此报告验证集表现。所选 checkpoint：<code>'+e(training['best_internal_validation_checkpoint'])+'</code>。</p>'
    report+='<section><h2>四项准确率</h2>'+table(['指标','Base','LoRA','变化（百分点）'],core)+'<p>失败输出按错误计入全部 400 条。Intensity 主指标仅计人工非空标签；三项同时正确要求 null 也匹配。</p></section>'
    report+='<section><h2>类别平衡指标</h2>'+table(['模型','指标','准确率','Macro F1','Balanced accuracy'],balanced)+'</section>'
    report+='<section><h2>少数类别与 null</h2>'+table(['模型','人工标签子集','准确率','Macro F1','Balanced accuracy'],subsets)+'</section>'
    report+='<section><h2>逐类表现</h2>'+table(['模型','指标','类别','人工条数','预测条数','Precision','Recall','F1'],classes)+'</section>'
    report+='<section><h2>输出质量与标签一致性</h2>'+table(['模型','有效输出','失败输出','六项全对','判决正确且内部一致'],quality)+'</section>'
    report+='<section><h2>原始结果和核验</h2><p class="links"><a href="comparison.json">comparison.json</a>'+''.join(links)+'</p><p>完整混淆矩阵、逐类计数、原始响应、重试 seed 和所有输入哈希保存在以上文件。JSON/传输错误可按固定预算恢复；语义错误不重试。</p></section></main></html>'
    (args.output_dir/'report.html').write_text(report,encoding='utf-8')
    print(json.dumps({'event':'validation_comparison_complete','items':400,'matched':True,'delta_pp':result['accuracy_delta_percentage_points'],'output_dir':str(args.output_dir)}))

if __name__=='__main__':main()
