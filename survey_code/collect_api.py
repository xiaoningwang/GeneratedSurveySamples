"""DeepSeek OpenAI-compatible requests with separate pilot and main outputs.

The runner never synthesises substitute API observations. Credentials are read
from an environment variable or a hidden terminal prompt and never persisted.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import argparse, os, json, hashlib, time, datetime, urllib.request, urllib.error, getpass, sys
import pandas as pd
from prepare_and_analyse import FEATURES, ROOT
from paths import DATA

QUESTIONS=(
    'Survey context: adults aged 18 or older in the United States in 2018. Recorded age 89 denotes age 89 or older. '
    'Answer the following ORIGINAL English survey items; do not change wording. '
    'Q1: It is much better for everyone involved if the man is the achiever outside the home '
    'and the woman takes care of the home and family. Answers: Strongly agree, Agree, Disagree, Strongly disagree. '
    'Q2: Taken all together, how would you say things are these days—would you say '
    'that you are very happy, pretty happy, or not too happy? Answers: Very happy, Pretty happy, Not too happy. '
)
ALLOWED={
    'gender':['Female','Male'],
    'education':['Less than high school','High school','College and above'],
    'marital_status':['Married','Single','Divorced/widowed'],
    'laborforce':['In the labor force','Not in the labor force'],
}


def prompts(mode,role,profile,train):
    text={
        'plain':'Complete the survey task.',
        'expert':'Act as a survey-data construction expert.',
        'respondent':'Act as a survey respondent.',
        'fewshot':'Complete the survey task using the supplied independent training examples.',
    }[role]+' '+QUESTIONS
    if role=='fewshot':
        # The original Windows run used CRLF inside the prompt's CSV examples.
        # Fix it explicitly so Linux/macOS reconstruct identical request hashes.
        text+=' Independent training data only (columns followed by rows): '+train[FEATURES+['agree','very_happy']].to_csv(index=False,lineterminator='\r\n')
        text+=' The released real data combine Strongly agree and Agree as agree=1, and Disagree and Strongly disagree as agree=0. very_happy=1 means Very happy; 0 combines the other two categories.'
    if mode=='conditional':
        text+=' Fixed respondent profile: '+json.dumps(profile,ensure_ascii=False)+'. Do not change these attributes. '
    else:
        text+=' First generate one adult respondent profile with gender, age, education, marital_status and laborforce. Allowed categories: '+json.dumps(ALLOWED)+'. Recorded age must be an integer 18–89, with 89 denoting age 89 or older. Do not use or infer benchmark answers. '
    text+=' Return only a JSON object with keys gender_role_attitude and happy, using the exact answer categories above.'
    if mode=='joint':
        text+=' Also include the five profile keys in the same JSON object.'
    return text


def validate(content,mode,profile):
    obj=json.loads(content)
    if not isinstance(obj,dict):raise ValueError('Expected a JSON object')
    if obj.get('gender_role_attitude') not in ['Strongly agree','Agree','Disagree','Strongly disagree']:
        raise ValueError('Invalid gender-role category')
    if obj.get('happy') not in ['Very happy','Pretty happy','Not too happy']:
        raise ValueError('Invalid happiness category')
    out=dict(profile) if mode=='conditional' else {k:obj[k] for k in FEATURES}
    if mode=='joint':
        for k,allowed in ALLOWED.items():
            if out[k] not in allowed:raise ValueError('Invalid profile '+k)
        age=out['age']
        if isinstance(age,bool) or not isinstance(age,(int,float)) or not 18<=age<=89 or int(age)!=age:
            raise ValueError('Invalid integer age')
    out.update(agree=int(obj['gender_role_attitude'] in ['Strongly agree','Agree']),
               very_happy=int(obj['happy']=='Very happy'),
               happy_category=obj['happy'],gender_role_category=obj['gender_role_attitude'])
    return out


def payload(model,prompt,thinking,temperature,max_tokens,effort):
    body={'model':model,'messages':[{'role':'user','content':prompt}],
          'response_format':{'type':'json_object'},'thinking':{'type':thinking},
          'max_tokens':max_tokens,'stream':False}
    if thinking=='disabled':body['temperature']=temperature
    else:body['reasoning_effort']=effort
    return body


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def get_key(ask=False):
    key=os.environ.get('DEEPSEEK_API_KEY','').strip()
    if not key and ask:
        if not sys.stdin.isatty():
            raise SystemExit('Please run in a normal terminal for hidden key entry, or configure DEEPSEEK_API_KEY locally.')
        key=getpass.getpass('DeepSeek API key（隐藏输入，不保存）: ').strip()
    if not key:raise SystemExit('未配置DEEPSEEK_API_KEY。请使用--ask-key在本地终端隐藏输入密钥。')
    return key


class StopAPI(RuntimeError):
    """Authentication, balance or permanent request errors stop the queue."""


def execute(job,cache,endpoint,headers,timeout,max_attempts,retry_transport=False):
    file=cache/(job['hash']+'.json')
    if file.exists():
        old=json.loads(file.read_text(encoding='utf-8'))
        retryable=old.get('status') in ['transport_error','http_transient_error']
        if not retry_transport or not retryable:
            return old
        history=cache/'failed_attempts';history.mkdir(exist_ok=True)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        file.replace(history/(job['hash']+'_'+stamp+'.json'))
    record={k:v for k,v in job.items() if k!='payload'}
    record.update(requested_model=job['payload']['model'],prompt=job['payload']['messages'][0]['content'],
                  timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),attempts=[])
    start=time.monotonic();response=None;stop_code=None
    for attempt in range(max_attempts):
        tick=time.monotonic()
        try:
            request=urllib.request.Request(endpoint,data=json.dumps(job['payload']).encode('utf-8'),headers=headers,method='POST')
            with urllib.request.urlopen(request,timeout=timeout) as r:
                response=json.load(r)
            record['attempts'].append({'attempt':attempt+1,'status':'response','seconds':time.monotonic()-tick})
            break
        except urllib.error.HTTPError as e:
            retryable=e.code in [408,429,500,502,503,504]
            record['attempts'].append({'attempt':attempt+1,'status':'http_error','code':e.code,'seconds':time.monotonic()-tick})
            record.update(status='http_transient_error' if retryable else 'http_error',error_code=e.code)
            if not retryable:
                stop_code=e.code;break
        except (TimeoutError,urllib.error.URLError,OSError):
            record['attempts'].append({'attempt':attempt+1,'status':'transport_error','seconds':time.monotonic()-tick})
            record['status']='transport_error'
        except (ValueError,TypeError):
            record['status']='invalid_response';break
        if attempt+1<max_attempts:time.sleep(min(2**attempt,8))
    record['seconds']=time.monotonic()-start
    if response is not None:
        if not isinstance(response,dict):
            record['status']='invalid_response';atomic_json(file,record);return record
        record.update(response=response,returned_model=response.get('model'),usage=response.get('usage',{}))
        try:
            choice=response['choices'][0]
            if choice.get('finish_reason')!='stop':raise ValueError('Non-stop finish reason')
            record['parsed']=validate(choice['message']['content'],job['mode'],job['profile'])
            record['status']='ok'
        except (ValueError,KeyError,TypeError,IndexError):record['status']='invalid_output'
    atomic_json(file,record)
    if stop_code is not None:
        raise StopAPI('API HTTP '+str(stop_code)+'：停止新增请求；请检查密钥、余额、权限或参数。已保存完成记录。')
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--model',default='deepseek-flash')
    p.add_argument('--base-url',default='https://api.deepseek.com')
    p.add_argument('--roles',nargs='+',choices=['plain','expert','respondent','fewshot'],default=['plain','expert','respondent','fewshot'])
    p.add_argument('--decode-repeats',type=int,default=3)
    p.add_argument('--joint-records',type=int,default=500)
    p.add_argument('--pilot',type=int,default=0)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--temperature',type=float,default=.7)
    p.add_argument('--thinking',choices=['enabled','disabled'],default='disabled')
    p.add_argument('--reasoning-effort',choices=['low','high','max'],default='high')
    p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--timeout',type=float,default=180)
    p.add_argument('--max-attempts',type=int,default=3)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--ask-key',action='store_true')
    p.add_argument('--retry-transport',action='store_true')
    a=p.parse_args()
    if min(a.decode_repeats,a.joint_records,a.workers,a.max_attempts,a.max_tokens)<1 or a.pilot<0 or not 0<=a.temperature<=2:
        raise SystemExit('Invalid count or temperature.')
    if len(a.roles)!=len(set(a.roles)):raise SystemExit('Roles must not be duplicated.')
    out=a.run_dir.resolve()/'results';out.mkdir(parents=True,exist_ok=True)
    cache=a.run_dir.resolve()/'api_raw';cache.mkdir(exist_ok=True)
    train=pd.read_csv(DATA/'training.csv');pop=pd.read_csv(DATA/'benchmark.csv')
    if set(train.profile_id)&set(pop.profile_id):raise SystemExit('Training and benchmark IDs overlap.')
    selected=pop.iloc[:a.pilot] if a.pilot else pop
    nj=min(a.joint_records,a.pilot) if a.pilot else a.joint_records
    jobs=[]
    for role in a.roles:
        for mode in ['conditional','joint']:
            iterator=selected.to_dict('records') if mode=='conditional' else [{'profile_id':i} for i in range(nj)]
            for row in iterator:
                profile={k:row[k] for k in FEATURES} if mode=='conditional' else {}
                for decode in range(a.decode_repeats if mode=='conditional' else 1):
                    body=payload(a.model,prompts(mode,role,profile,train),a.thinking,a.temperature,a.max_tokens,a.reasoning_effort)
                    identity={'body':body,'decode':decode,'record':int(row['profile_id']),'mode':mode,'role':role}
                    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
                    jobs.append(dict(hash=digest,mode=mode,role=role,decode=decode,profile_id=int(row['profile_id']),profile=profile,payload=body))
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ['ask_key','retry_transport','preflight','run_dir']}
    config.update(planned_requests=len(jobs),benchmark_rows=len(pop),training_rows=len(train),
                  planned_hashes=[j['hash'] for j in jobs],
                  data_sha256={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['training.csv','benchmark.csv']},
                  prompt_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    plan_id=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    atomic_json(out/('api_plan_'+plan_id+'.json'),config)
    if (out/'api_plan.json').exists():
        old=json.loads((out/'api_plan.json').read_text(encoding='utf-8'))
        identity_fields=['planned_hashes','data_sha256','prompt_code_sha256','model','thinking','pilot']
        if any(old.get(k)!=config.get(k) for k in identity_fields):raise SystemExit('Run configuration changed. Use a new run directory rather than overwrite existing evidence.')
    print(f'计划请求 {len(jobs)}；模式：'+('试运行' if a.pilot else '正式实验'),flush=True)
    if a.preflight:
        atomic_json(out/'api_preflight.json',config)
        print('仅检查队列，没有发送API请求。');return
    key=get_key(a.ask_key)
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    models_request=urllib.request.Request(a.base_url.rstrip('/')+'/models',headers=headers)
    try:
        with urllib.request.urlopen(models_request,timeout=30) as r:available=json.load(r)
    except urllib.error.HTTPError as e:raise SystemExit('模型列表验证失败，HTTP '+str(e.code)+'。请检查密钥、权限和余额。') from None
    except (urllib.error.URLError,TimeoutError,OSError):raise SystemExit('无法连接模型列表。请检查网络。') from None
    if a.model not in [m['id'] for m in available.get('data',[])]:
        raise SystemExit('所选模型不在接口返回的列表中。请明确更改config.json中的models，不会自动替换模型。')
    atomic_json(out/'models_available.json',available)
    endpoint=a.base_url.rstrip('/')+'/chat/completions'
    rows=[];complete=False
    def save():
        if rows:
            df=pd.DataFrame(rows);tmp=out/'api_records.csv.tmp';df.to_csv(tmp,index=False,encoding='utf-8-sig');tmp.replace(out/'api_records.csv')
        atomic_json(out/'api_plan.json',config)
        atomic_json(out/'api_run_status.json',dict(api_requests_planned=len(jobs),records=len(rows),valid=sum(r['status']=='ok' for r in rows),
                    pilot=bool(a.pilot),requested_model=a.model,plan_id=plan_id,complete_queue=complete))
    pool=ThreadPoolExecutor(max_workers=a.workers);pending=set();iterator=iter(jobs)
    def submit_one():
        job=next(iterator,None)
        if job is None:return False
        pending.add(pool.submit(execute,job,cache,endpoint,headers,a.timeout,a.max_attempts,a.retry_transport));return True
    try:
        for _ in range(a.workers):submit_one()
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                record=future.result()
                row={k:record.get(k) for k in ['hash','mode','role','decode','profile_id','status','returned_model','seconds']}
                row.update(record.get('parsed',{}));rows.append(row)
                if len(rows)%20==0:
                    save();print(f'已完成 {len(rows)}/{len(jobs)}，有效 {sum(r["status"]=="ok" for r in rows)}',flush=True)
                submit_one()
        complete=True
    except KeyboardInterrupt:
        print('已中断。原始响应已保存；重新运行同一编号脚本可以续跑。',flush=True)
        for f in pending:f.cancel()
        raise SystemExit(130)
    except StopAPI as e:
        for f in pending:f.cancel()
        raise SystemExit(str(e)) from None
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
        save()
    print(f'完成队列：{len(rows)}，有效：{sum(r["status"]=="ok" for r in rows)}。原始输出目录：{cache}',flush=True)


if __name__=='__main__':main()
