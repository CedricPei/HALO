"""Wait for a completed LoRA, then evaluate LoRA and base on 400 validation rows.

Only this job's verified GPU1 serving processes are stopped. Runs preserve raw
HTTP responses, exact inference settings and a single structural-recovery pass.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
from .lora_job import gpu_state, gpu_admitted
from .post_training_job import _owned_server, _proc, stop_owned_server
from .runner import _get_api_key, _parse_response, _read_jsonl, _rewrite_jsonl
from .training_job import wait_serving


def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):digest.update(b)
    return digest.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    temporary=Path(p).with_name(Path(p).name+'.tmp')
    try:temporary.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');temporary.replace(p)
    finally:temporary.unlink(missing_ok=True)


def repair_raw(folder):
    path=folder/'predictions.jsonl'
    predictions={r['item_id']:r for r in _read_jsonl(path,recover_partial=True)}
    repaired=0
    for raw in _read_jsonl(folder/'raw_responses.jsonl',recover_partial=True):
        attempts=raw.get('attempts',[])
        if not attempts:continue
        last=attempts[-1];prior=predictions.get(raw['item_id'])
        if prior and prior.get('finished_utc','')>=last.get('finished_utc',''):continue
        record={k:raw.get(k) for k in ['item_id','messages_sha256','input_context_sha256','invocation_id','prior_attempts']}
        record.update(status=last['status'],prediction=None,usage=last.get('usage',{}),attempts=len(attempts),
            cumulative_attempts=raw.get('prior_attempts',0)+len(attempts),started_utc=attempts[0]['started_utc'],finished_utc=last['finished_utc'],recovered_from_persisted_raw=True)
        for k in ['error','seed','seed_ordinal','request_sha256','finish_reason','response_id','diagnostic_consistency','diagnostic_consistency_error']:
            if k in last:record[k]=last[k]
        if last['status']=='ok':record.update(_parse_response(last['response']))
        predictions[raw['item_id']]=record;repaired+=1
    if repaired:_rewrite_jsonl(path,list(predictions.values()))


def run(path):
    config=read(path);cwd=Path(config['cwd']);output=Path(config['job_output_dir'])
    output.mkdir(parents=True,exist_ok=True);state_path=output/'job.json';child=None
    with (output/'job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(state_path) if state_path.exists() else {'status':'waiting_for_training','created_utc':utc(),'config_sha256':sha(path),'completed_stages':[]}
        if state['config_sha256']!=sha(path):raise ValueError('Frozen evaluation launch changed')
        if state['status']=='completed':return
        if state.get('child_pid') and _proc(state['child_pid']):raise RuntimeError('Previous stage child is still alive')
        stat=Path('/proc/'+str(os.getpid())+'/stat').read_text()
        state.update(supervisor_pid=os.getpid(),supervisor_uid=os.getuid(),supervisor_start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),resumed_utc=utc(),supervisor_source_sha256=sha(__file__))
        def persist():state['updated_utc']=utc();write(state_path,state)
        def phase(name):state['status']=name;persist();print(json.dumps({'event':'stage','status':name,'utc':utc()}),flush=True)
        def stage_done(name):
            if name not in state['completed_stages']:state['completed_stages'].append(name)
            persist()
        def execute(command,environment=None):
            nonlocal child
            child=subprocess.Popen(command,cwd=cwd,env=environment,stdin=subprocess.DEVNULL,start_new_session=True)
            state.update(child_pid=child.pid,child_command=command);persist()
            while child.poll() is None:
                serving_pid=state.get('serving_pid')
                if serving_pid and _proc(serving_pid) is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise RuntimeError('Owned serving process exited during evaluation; stage stopped')
                time.sleep(2)
            code=child.returncode;state.update(child_pid=None,child_exit_code=code);persist();child=None
            if code:raise RuntimeError(f"Stage {state['status']} returned {code}")
        def cancelled(signum,frame):raise KeyboardInterrupt(f'Received signal {signum}')
        signal.signal(signal.SIGTERM,cancelled);signal.signal(signal.SIGINT,cancelled)
        def stop_server():
            pid=state.get('serving_pid')
            if pid and _proc(pid):
                st=Path('/proc/'+str(pid)+'/stat').read_text();f=st[st.rfind(')')+2:].split()
                if int(f[19])!=state['serving_start_ticks']:raise RuntimeError('Serving PID was reused')
                stop_owned_server(pid,state['serving_model_path'],config['serving'],allow_foreign_gpu_processes=config.get('gpu_admission',{}).get('mode')=='shared')
            state.update(serving_pid=None,serving_model_path=None);persist()
        def serve(label,model,manifest):
            pid=state.get('serving_pid')
            if pid and _proc(pid):
                if state.get('serving_model_path')!=str(model):stop_server()
                else:_owned_server(pid,str(model),config['model']);return
            phase('waiting_for_gpu_'+label)
            admission=config.get('gpu_admission',{'mode':'exclusive'})
            while not gpu_admitted(gpu_state(config['gpu_index'],config['gpu_uuid']),admission):time.sleep(5)
            with socket.socket() as port_check:
                port_check.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                port_check.bind(('127.0.0.1',config['port']))
            env=os.environ.copy();env.update(config['serving_environment'])
            env['VLLM_API_KEY']=_get_api_key(config['api_key_env'],Path(config['env_file']))
            env.pop('PYTHONPATH',None)
            command=config['serving_command_prefix']+['serve',str(model)]+config['serving_arguments']
            phase('starting_'+label+'_server')
            with (output/('serve_'+label+'.log')).open('ab',buffering=0) as log:
                process=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            stat=Path('/proc/'+str(process.pid)+'/stat').read_text()
            state.update(serving_pid=process.pid,serving_model_path=str(model),serving_start_ticks=int(stat[stat.rfind(')')+2:].split()[19]));persist()
            readiness=wait_serving(process,{'restore_timeout_seconds':600,'restore_base_url':config['base_url'],'restore_model_id':config['model'],'restore_env_file':config['env_file'],'restore_api_key_env':config['api_key_env']})
            state['serving_readiness']=readiness;persist()
            if readiness['status']!='ready':raise RuntimeError(label+' serving failed readiness: '+readiness['status'])
            _owned_server(process.pid,str(model),config['model'])
        def evaluate(label,manifest):
            folder=Path(config['run_dirs'][label]);folder.mkdir(parents=True,exist_ok=True)
            manifest_path=output/('serving_'+label+'.json')
            if manifest_path.exists() and read(manifest_path)!=manifest:raise ValueError('Serving lineage changed')
            if not manifest_path.exists():write(manifest_path,manifest)
            base=[config['serving_python'],'-B','-m','evaluation.judge_bench.runner','--input',config['validation_input'],'--output-dir',str(folder),'--base-url',config['base_url'],'--model',config['model'],'--api-key-env',config['api_key_env'],'--env-file',config['env_file'],'--server-config',str(manifest_path),'--gpu-index',str(config['gpu_index']),'--record-latency','false']
            for name,value in config['inference'].items():base.extend(['--'+name,str(value)])
            first=label+'_first_pass'
            if first not in state['completed_stages']:
                phase('evaluating_'+label+'_400');repair_raw(folder)
                execute(base+['--max-structure-retries','0'])
                first_dir=folder/'first_pass';first_dir.mkdir(exist_ok=True)
                for name in ['metrics.json','predictions.jsonl','run_metadata.json']:shutil.copyfile(folder/name,first_dir/name)
                stage_done(first)
            if read(folder/'metrics.json')['failed_predictions'] and label+'_recovery' not in state['completed_stages']:
                phase('recovering_'+label+'_failed_ids');repair_raw(folder)
                invocations=read(folder/'run_metadata.json').get('retry_invocations',[])
                started=any(i.get('mode')=='retry_failed' for i in invocations)
                execute(base+['--max-structure-retries','2']+([] if started else ['--retry-failed']))
                stage_done(label+'_recovery')
            meta=read(folder/'run_metadata.json')
            if not meta.get('complete') or meta['completed_predictions']!=400:raise RuntimeError(label+' evaluation is incomplete')
            stage_done(label+'_evaluated')
        try:
            persist();training_dir=Path(config['training_output_dir'])
            while True:
                train_job=read(training_dir/'job.json')
                if train_job['status']=='completed':break
                if train_job['status'] in ['failed','cancelled']:raise RuntimeError('Training '+train_job['status'])
                time.sleep(5)
            training=read(training_dir/'training_metadata.json')
            if training['status']!='complete' or training.get('full_epoch_training_complete') is not True:raise RuntimeError('Final LoRA training is not complete')
            for name,expected in config['frozen_sha256'].items():
                if sha(name)!=expected:raise ValueError('Frozen evaluation dependency changed: '+name)
            state.update(training_finished_utc=training['finished_utc'],selected_checkpoint=training['best_internal_validation_checkpoint']);persist()
            merged=Path(config['merged_model_dir'])
            if 'merged' not in state['completed_stages']:
                phase('merging_adapter_cpu');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
                counts=config.get('model_tensor_counts',{'base':775,'mtp':15})
                execute([config['training_python'],'-B','-m','evaluation.judge_bench.merge_adapter','--base-model',config['base_model_dir'],'--adapter',str(training_dir/'adapter'),'--training-metadata',str(training_dir/'training_metadata.json'),'--training-launch-config',config['training_launch_config'],'--base-manifest',config['base_artifact_manifest'],'--output-dir',str(merged),'--cpu-threads','8','--expected-base-keys',str(counts['base']),'--expected-mtp-keys',str(counts['mtp']),'--resume'],env)
                stage_done('merged')
            merge=read(merged/'merge_manifest.json')
            original=read(config['base_artifact_manifest'])
            versions={name:importlib.metadata.version(name) for name in config['serving']['runtime_versions']}
            if versions!=config['serving']['runtime_versions']:raise ValueError('Serving runtime versions differ from the declared runtime')
            base_manifest=copy.deepcopy(original);base_manifest.update(config['serving'])
            for key in ['launcher_path','launcher_sha256','original_launcher_sha256']:base_manifest.pop(key,None)
            base_manifest.update(evaluated_model_name=config['model'],evaluation_scope='400-item checkpoint-selection validation set',captured_utc=state['created_utc'],artifact_evidence_captured_utc=original['captured_utc'],launch_config_path=str(path),launch_config_sha256=state['config_sha256'])
            lora_manifest=copy.deepcopy(base_manifest);lora_manifest.update(model_path=str(merged),model_artifacts=merge['model_artifacts'],adaptation={k:v for k,v in merge.items() if k!='model_artifacts'})
            for label,model,manifest in [('lora',merged,lora_manifest),('base',Path(config['base_model_dir']),base_manifest)]:
                if label+'_evaluated' in state['completed_stages']:continue
                serve(label,model,manifest);evaluate(label,manifest);phase('stopping_'+label+'_server');stop_server()
            phase('writing_validation_comparison')
            execute([config['serving_python'],'-B','-m','evaluation.judge_bench.validation_comparison','--input',config['validation_input'],'--base-run-dir',config['run_dirs']['base'],'--lora-run-dir',config['run_dirs']['lora'],'--training-metadata',str(training_dir/'training_metadata.json'),'--output-dir',config['comparison_dir']])
            state.update(status='completed',finished_utc=utc(),comparison_dir=config['comparison_dir']);persist()
        except KeyboardInterrupt as error:
            if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            stop_server();state.update(status='cancelled',error=str(error),finished_utc=utc());persist()
        except Exception as error:
            state.update(status='failed',error=f'{type(error).__name__}: {error}',finished_utc=utc());persist();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);run(p.parse_args().config)
if __name__=='__main__':main()
