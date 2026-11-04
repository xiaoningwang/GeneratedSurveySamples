import json
import numpy as np
import pandas as pd
from paths import OUT,DETAILS,MECHANISM,API

def save():
    rows=[]
    def add(part,item,condition,value,unit=''):
        if pd.notna(value):rows.append({'部分':part,'指标':item,'条件':condition,'数值':float(value),'单位':unit})
    sim=json.loads((MECHANISM/'simulation_results.json').read_text(encoding='utf-8'))
    for s in sim['scenarios']:
        for key,label,unit in [('truth','总体均值','比例'),('variance_reduction_pct','残差校正方差降低比例','%')]:add('模拟研究',label,s['name'],s[key],unit)
        add('模拟研究','联合分组校准理论偏差',s['name'],np.dot(s['p'],s['b']),'比例')
    add('模拟研究','稀有群体期望真实样本量','总体三',sim['n']*sim['scenarios'][2]['p'][3],'条')
    mcse=np.sqrt(.95*.05/sim['R'])*100
    add('评价方法','覆盖率蒙特卡洛标准误','名义95%，R=2000',mcse,'百分点')
    add('评价方法','近似95%蒙特卡洛误差半宽','名义95%，R=2000',1.96*mcse,'百分点')
    specs=[(DETAILS/'group_gap_mc.csv','群体差值模拟',['population','method'],{'bias_on_valid':('差值偏差',1,'比例'),'coverage_all_repeats':('覆盖率',100,'%'),'failure_rate':('区间失败率',100,'%')}),
           (DETAILS/'sample_sizes_mc.csv','模拟样本量',['n','m','method'],{'bias':('偏差',1,'比例'),'coverage':('覆盖率',100,'%'),'width':('平均区间宽度',1,'比例')}),
           (API/'api_quality.csv','实证分布',['role','target'],{'real_mean':('真实比例',1,'比例'),'conditional_mean':('首次条件比例',1,'比例'),'joint_mean':('联合生成比例',1,'比例'),'structure_bias':('样本结构贡献',1,'比例'),'answer_bias':('条件回答贡献',1,'比例'),'real_sex_gap':('真实性别差值',1,'比例'),'generated_sex_gap':('生成性别差值',1,'比例')}),
           (API/'api_calibration.csv','实证校准',['role','target','method'],{'estimate':('校准比例',1,'比例'),'effective_generated_n':('有效生成样本量',1,'条')}),
           (API/'api_generation_sizes.csv','联合生成样本量',['role','target','m'],{'raw_mean':('联合生成比例',1,'比例')})]
    for path,part,keys,metrics in specs:
        for _,z in pd.read_csv(path).iterrows():
            condition=', '.join(str(z[k]) for k in keys)
            for key,(name,scale,unit) in metrics.items():add(part,name,condition,z.get(key,np.nan)*scale,unit)
    inf=pd.read_csv(API/'api_inference.csv')
    for _,z in inf[(inf.role=='fewshot')&inf.target.isin(['agree','very_happy'])&(inf.method=='大模型残差校正')].iterrows():
        condition=f'{z.target}, n={z.n}'
        add('实证样本量','均方根误差',condition,z.rmse,'比例')
        add('实证样本量','平均区间宽度',condition,z.width,'比例')
        add('实证样本量','估计方差增加比例',condition,-100*z.variance_reduction_vs_n,'%')
    pd.DataFrame(rows).to_csv(OUT/'正文数值.csv',index=False,encoding='utf-8-sig')
