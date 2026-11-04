"""Run the original API collection algorithm with the original configuration."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parent.parent


def run(pilot=False):
    parser=argparse.ArgumentParser(description='原实验大语言模型数据采集入口')
    parser.add_argument('--dry-run',action='store_true',help='仅生成请求队列，不调用API。')
    parser.add_argument('--retry-transport',action='store_true',help='仅重试传输类失败，不重抽有效或格式失败回答。')
    args=parser.parse_args()
    c=json.loads((ROOT/'api_config.json').read_text(encoding='utf-8-sig'))
    saved=json.loads((ROOT/'survey_data/api_metadata.json').read_text(encoding='utf-8-sig'))
    expected={'base_url':saved['base_url'],'models':[saved['model']],'roles':saved['roles'],'decode_repeats':saved['decode_repeats'],'joint_records':saved['joint_records'],
              'thinking':saved['thinking'],'reasoning_effort':saved['reasoning_effort'],'temperature':saved['temperature'],'max_tokens':saved['max_tokens'],
              'workers':saved['workers'],'timeout_seconds':saved['timeout'],'max_attempts':saved['max_attempts']}
    for key,value in expected.items():
        if c[key]!=value:raise SystemExit(f'api_config.json中的{key}与原论文实验不一致。')
    if c['pilot_profiles']!=4 or c['pilot_decode_repeats']!=1:raise SystemExit('原试运行使用4个画像、每画像1次解码。')
    folder=ROOT/'paper_results/llm_collection'/('pilot' if pilot else 'main')
    command=[sys.executable,str(ROOT/'survey_code/collect_api.py'),'--run-dir',str(folder),'--model',c['models'][0],'--base-url',c['base_url'],
             '--roles',*c['roles'],'--decode-repeats',str(c['pilot_decode_repeats'] if pilot else c['decode_repeats']),
             '--joint-records',str(c['joint_records']),'--workers',str(c['workers']),'--thinking',c['thinking'],'--reasoning-effort',c['reasoning_effort'],
             '--temperature',str(c['temperature']),'--max-tokens',str(c['max_tokens']),'--timeout',str(c['timeout_seconds']),'--max-attempts',str(c['max_attempts'])]
    if pilot:command+=['--pilot',str(c['pilot_profiles'])]
    if args.dry_run:command+=['--preflight']
    else:command+=['--ask-key']
    if args.retry_transport:command+=['--retry-transport']
    print('试运行计划32次请求。' if pilot else '正式计划8708次请求。',flush=True)
    print('采集输出目录：'+str(folder),flush=True)
    subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':run()
