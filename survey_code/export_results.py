"""Save all fourteen tables and five figures to paper_results."""
from pathlib import Path
import argparse,csv,json,re,shutil,sys
import numpy as np
import pandas as pd
from concept_tables import CONCEPT_TABLES
from paths import ROOT,DATA,OUT,DETAILS,MECHANISM,API

OUTPUT=OUT
ROLES={'普通指令':'plain','调查专家提示':'expert','受访者角色提示':'respondent','少样本提示':'fewshot'}
TITLES={1:'信息来源及其统计作用',2:'领域微调前后的统计检验通过率',3:'评价对象指标与判定依据',4:'四种已知总体的模拟参数',5:'不同总体下六种方法的模拟结果',6:'联合分组校准均值偏差与群体差值的对应关系',7:'总体一的样本量敏感性结果',8:'真实调查评价有限总体的基本特征',9:'生成实验与统计方法对照',10:'真实调查基准上的统计方法对照',11:'四种提示下条件回答与联合生成的实际结果',12:'重复回答平均的预测质量',13:'实际预测下的均值推断与真实标签预算比较',14:'少样本提示下预设群体差值的推断结果'}
# zero-based column -> number of decimal places displayed in the manuscript
NUMERIC={2:{1:1,2:1,3:1},4:{i:2 for i in range(2,6)},5:{2:4,3:4,4:2,5:4},6:{i:4 for i in range(1,5)},7:{3:4,4:2,5:4},8:{1:0,2:4,3:4},10:{2:0,3:3,4:2},11:{i:3 for i in range(2,7)},12:{i:3 for i in range(2,6)},13:{2:0,3:3,4:3,5:2,6:3},14:{2:3,3:3,4:2,5:3}}




def one(frame,**filters):
    for key,value in filters.items():frame=frame[frame[key]==value]
    if len(frame)!=1:raise ValueError(f'结果筛选必须唯一：{filters}；实际{len(frame)}行。')
    return frame.iloc[0]


def calculate_tables():
    read=lambda p:pd.read_csv(DETAILS/p)
    tables={}
    published=json.loads((DATA/'published_inputs.json').read_text(encoding='utf-8'))
    tables[2]=[['评价数据集','微调前（%）','微调后（%）','提高幅度（百分点）']]
    from decimal import Decimal
    for a in published['xie_table1']:
        before,after=Decimal(a['before_pct']),Decimal(a['after_pct'])
        tables[2].append([a['dataset'].replace('现期人口调查','当前人口调查'),float(before),float(after),float(after-before)])
    sim=json.loads((MECHANISM/'simulation_results.json').read_text(encoding='utf-8'))
    tables[4]=[['总体','参数','(0,0)组','(0,1)组','(1,0)组','(1,1)组']]
    for scenario in sim['scenarios']:
        for key,label in [('p','目标组比例'),('q','生成组比例'),('mu','真实组均值'),('b','条件回答偏差')]:
            tables[4].append([scenario['name'].replace('总体',''),label]+scenario[key])
    tables[5]=[['总体','方法','偏差','均方根误差','覆盖率（%）','平均宽度']]
    for scenario in sim['scenarios']:
        for m in scenario['methods']:
            label=m['method'].replace(' ','（')+('）' if ' ' in m['method'] else '')
            tables[5].append([scenario['name'].replace('总体',''),label,m['bias'],m['rmse'],100*m['coverage'],m['width']])
    z=read('group_gap_theory.csv')
    tables[6]=[['总体','均值偏差','真实群体差值','合成群体差值','差值偏差']]+z[['population','mean_bias_joint','true_gap','generated_gap','gap_bias']].values.tolist()
    z=read('sample_sizes_mc.csv');tables[7]=[['真实样本量n','生成样本量m','方法','偏差','覆盖率（%）','平均宽度']]
    for n,m,method in [(200,500,'联合分组校准'),(200,2000,'联合分组校准'),(200,5000,'联合分组校准'),(50,5000,'残差校正'),(200,5000,'残差校正'),(500,5000,'残差校正')]:
        a=one(z,n=n,m=m,method=method);tables[7].append([n,m,method,a.bias,100*a.coverage,a.width])
    z=read('real_benchmark.csv');tables[8]=[['群体','记录数','传统分工同意比例','非常幸福比例']]+z[['group','N','agree','very_happy']].values.tolist()
    controls=read('real_paired_controls.csv');tables[10]=[['目标','方法','真实标签数','均方根误差','覆盖率（%）']]
    for target,label in [('agree','传统分工同意'),('very_happy','非常幸福')]:
        for display,method in [('直接估计','真实样本均值'),('逻辑回归残差校正','logistic残差校正'),('随机森林残差校正','forest残差校正'),('相同真实标签预算直接估计','等标签预算直接估计')]:
            a=one(controls,target=target,n=200,method=method);tables[10].append([label,display,a.label_budget,a.rmse,100*a.coverage])
    api=API;quality=pd.read_csv(api/'api_quality.csv');pred=pd.read_csv(api/'api_prediction_controls.csv');infer=pd.read_csv(api/'api_inference.csv')
    tables[11]=[['提示','题项','真实比例','条件比例','条件回答偏差','联合比例','回答方差比']]
    for display,role in ROLES.items():
        for target,label in [('agree','传统分工'),('very_happy','非常幸福')]:
            a=one(quality,role=role,target=target);tables[11].append([display,label]+[a[k] for k in ['real_mean','conditional_mean','conditional_bias','joint_mean','answer_variance_ratio']])
    tables[12]=[['题项','预测方法','预测均值','Brier分数','AUC','校准误差']]
    for target,label in [('agree','传统分工'),('very_happy','非常幸福')]:
        for display,role in list(ROLES.items())+[('逻辑回归','fewshot'),('随机森林','fewshot')]:
            model={'逻辑回归':'logistic','随机森林':'forest'}.get(display,'大模型')
            a=one(pred,role=role,target=target,model=model);tables[12].append([label,display]+[a[k] for k in ['prediction_mean','brier','auc','ece_10_fixed_bins']])
    tables[13]=[['题项','方法','标签数','偏差','均方根误差','覆盖率%','区间宽度']]
    for target,label in [('agree','传统分工'),('very_happy','非常幸福')]:
        items=[(name+'校正',role,'大模型残差校正') for name,role in ROLES.items()]
        items += [('直接估计','fewshot','真实样本均值'),('相同真实标签预算直接估计','fewshot','大模型等标签预算直接估计'),('逻辑回归生成校正','fewshot','logistic_generated残差校正')]
        for display,role,method in items:
            a=one(infer,role=role,target=target,n=200,method=method);tables[13].append([label,display,a.label_budget,a.bias,a.rmse,100*a.coverage,a.width])
    tables[14]=[['题项','群体差值','直接估计均方根误差','校正均方根误差','校正覆盖率%','校正区间宽度']]
    contrasts=[('女性减男性','female_minus_male'),('高龄减低龄','60plus_minus_under60'),('高龄女性减低龄男性','female_60plus_minus_male_under60')]
    for target,label in [('agree','传统分工'),('very_happy','非常幸福')]:
        for name,suffix in contrasts:
            key=target+'_'+suffix;a=one(infer,role='fewshot',target=key,n=200,method='真实群体差值');b=one(infer,role='fewshot',target=key,n=200,method='大模型群体残差校正')
            tables[14].append([label,name,a.rmse,b.rmse,100*b.coverage,b.width])
    return tables


def normal(text):return re.sub(r'\s+','',str(text)).replace('−','-')






def export_tables():
    tables=calculate_tables();target=OUTPUT;target.mkdir(parents=True,exist_ok=True)
    index=[]
    for number in range(1,15):
        rows=tables.get(number,CONCEPT_TABLES.get(number));formatted=[]
        for i,row in enumerate(rows):
            values=[]
            for j,value in enumerate(row):
                if i and j in NUMERIC.get(number,{}):value=f'{float(value):.{NUMERIC[number][j]}f}'
                elif isinstance(value,(int,np.integer)):value=str(value)
                elif isinstance(value,float) and value.is_integer():value=str(int(value))
                values.append(str(value))
            formatted.append(values)
        path=target/f'table_{number:02d}.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as stream:csv.writer(stream).writerows(formatted)
        index.append({'table':number,'title':TITLES[number],'file':path.name})
    (OUTPUT/'table_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    return tables


def export_figures(offline=False):
    destination=OUTPUT;destination.mkdir(parents=True,exist_ok=True)
    pairs=[(MECHANISM/'figure1_coverage.png','figure_01.png'),(DETAILS/'group_gap.png','figure_02.png'),(DETAILS/'sample_sizes.png','figure_03.png'),(DETAILS/'api_workflow.png','figure_04.png')]
    for src,dst in pairs:shutil.copyfile(src,destination/dst)
    if offline:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties,fontManager
    font=FontProperties(fname=str(ROOT/'survey_code/fonts/NotoSerifCJKsc-Regular.otf'));fontManager.addfont(font.get_file())
    with plt.rc_context({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':8.5}):
        data=pd.read_csv(API/'api_inference.csv');data=data[(data.role=='fewshot')&(data.target=='agree')]
        items=[('真实样本均值','直接估计 n','o-'),('大模型残差校正','少样本提示残差校正','s--'),('logistic残差校正','逻辑回归残差校正','^:'),('大模型等标签预算直接估计','相同真实标签预算\n直接估计（n+100）','D-.')]
        fig,axes=plt.subplots(1,2,figsize=(8.0,3.4));curve_rows=[]
        for method,label,style in items:
            block=data[data.method==method].sort_values('n')
            if block.n.tolist()!=[50,100,200]:raise AssertionError('论文图5缺少预定样本量。')
            for ax,col,scale in [(axes[0],'rmse',1),(axes[1],'coverage',100)]:ax.plot(block.n,block[col]*scale,style,color='black',lw=.9,label=label,ms=4)
            curve_rows.extend(block[['n','method','rmse','coverage','label_budget']].to_dict('records'))
        axes[0].set_ylabel('均方根误差（RMSE）');axes[1].set_ylabel('区间覆盖率（%）');axes[1].axhline(95,color='gray',lw=.7)
        for ax in axes:ax.set_xlabel('真实校正样本量 n');ax.set_xticks([50,100,200])
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=2,frameon=False,fontsize=7.1,handlelength=3)
        fig.tight_layout(rect=(0,0,1,.80));fig.savefig(destination/'figure_05.png',dpi=400);fig.savefig(destination/'figure_05.pdf');plt.close(fig)
        pd.DataFrame(curve_rows).to_csv(OUTPUT/'figure_05_data.csv',index=False,encoding='utf-8-sig')
    index=[{'figure':i,'file':f'figure_{i:02d}.png'} for i in range(1,6)]
    (OUTPUT/'figure_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    print('论文图1—图5已按现稿编号导出；图5固定展示4种方法。')


