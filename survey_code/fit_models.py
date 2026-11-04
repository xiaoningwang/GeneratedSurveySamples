from paths import ROOT,DATA,DETAILS
import hashlib,json

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    c={'decode_repeats':3,'mc_repeats':2000}
    import pandas as pd
    import numpy as np
    from prepare_and_analyse import fit_controls, finite_population_check
    from paired_evaluation import paired_inference, quality_controls
    train=pd.read_csv(DATA/'training.csv');pop=pd.read_csv(DATA/'benchmark.csv')
    fit_controls(train,pop)
    pred=pd.read_csv(DETAILS/'statistical_predictions.csv')
    # Match the number of categorical generations used to approximate LLM probabilities.
    # The analytic statistical probabilities are retained as a separate benchmark.
    rng=np.random.default_rng(20261023);sampled=pred[['profile_id']].copy();generation=[]
    for model in ['logistic','forest','cart']:
        draws={}
        for target in ['agree','very_happy']:
            prob=pred[model+'_'+target].to_numpy(float)
            draws[target]=rng.binomial(1,prob[:,None],size=(len(pop),c['decode_repeats']))
            sampled[model+'_generated_'+target]=draws[target].mean(1)
        for i,profile_id in enumerate(pop.profile_id):
            for k in range(c['decode_repeats']):
                generation.append({'model':model,'profile_id':int(profile_id),'decode':k,
                                   'agree':int(draws['agree'][i,k]),'very_happy':int(draws['very_happy'][i,k]),
                                   'data_origin':'statistical_probability_generation_not_api'})
    sampled.to_csv(DETAILS/'statistical_sampled_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(generation).to_csv(DETAILS/'statistical_generation.csv',index=False,encoding='utf-8-sig')
    (DETAILS/'statistical_generation_manifest.json').write_text(json.dumps({'decode_repeats':c['decode_repeats'],'seed':20261023,
        'benchmark_hash':digest(DATA/'benchmark.csv'),'statistical_prediction_hash':digest(DETAILS/'statistical_predictions.csv'),
        'targets_generated_independently_given_profile':True},indent=2),encoding='utf-8')
    finite_population_check(pop,pred,c['mc_repeats'])
    paired=[];quality=[]
    for target in ['agree','very_happy']:
        values={m:pred[m+'_'+target].to_numpy() for m in ['logistic','forest','cart']}
        values.update({m+'_generated':sampled[m+'_generated_'+target].to_numpy() for m in ['logistic','forest','cart']})
        paired+=paired_inference(pop,values,target,'fewshot',c['mc_repeats'])
        quality+=quality_controls(pop,values,target,'statistical_controls')
    frame=pd.DataFrame(paired);frame['method']=frame['method'].str.replace('大模型等标签预算直接估计','等标签预算直接估计',regex=False)
    frame.to_csv(DETAILS/'real_paired_controls.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(quality).to_csv(DETAILS/'statistical_quality.csv',index=False,encoding='utf-8-sig')
    print('统计模型及预算对照计算完成。')
