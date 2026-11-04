"""Evaluate actual cached calls. Requires full conditional response coverage.

Repeated finite-population sampling evaluates a FROZEN predictor, not an
unconditional guarantee over API retraining or future model versions.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import brier_score_loss,roc_auc_score
from prepare_and_analyse import ROOT,SEED,TARGETS,finite_population_check
from paths import DATA,DETAILS,API
from paired_evaluation import paired_inference,quality_controls,group_descriptions

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--R',type=int,default=2000);args=parser.parse_args()
    OUT=API
    OUT.mkdir(parents=True,exist_ok=True)
    path=DATA/'api_records.csv'
    if not path.exists():raise SystemExit('No actual API records. Run run_deepseek.py first; no mock estimates will be generated.')
    # Read the archived collection log rather than assume the queue completed.
    status=json.loads((DATA/'api_raw/main_api_run_status.json').read_text(encoding='utf-8-sig'))
    if status.get('pilot'):raise SystemExit('Pilot outputs cannot be used as full manuscript results. Complete the main queue first.')
    calls=pd.read_csv(path);pop=pd.read_csv(DATA/'benchmark.csv');pop=pop.set_index('profile_id',drop=False)
    plan=json.loads((DATA/'api_metadata.json').read_text());K=plan['decode_repeats']
    if not status.get('complete_queue'):raise SystemExit('Formal API queue is incomplete. Resume 05_api_formal.py first.')
    if 'planned_hashes' not in plan:raise SystemExit('Missing immutable request manifest; rerun the current API runner before evaluation.')
    if set(calls.hash)!=set(plan['planned_hashes']) or calls.hash.duplicated().any():raise SystemExit('Actual records do not match the full request manifest.')
    for name,expected in plan['data_sha256'].items():
        if hashlib.sha256((DATA/name).read_bytes()).hexdigest()!=expected:raise SystemExit('Input data changed since the API run.')
    versions=calls.loc[calls.status=='ok','returned_model'].dropna().unique()
    if len(versions)!=1:raise SystemExit('A single reported response model is required; mixed or missing versions need separate experiments.')
    stats=pd.read_csv(DETAILS/'statistical_predictions.csv').set_index('profile_id').reindex(pop.index)
    if stats.isna().any().any():raise SystemExit('Statistical control predictions are incomplete.')
    generation_manifest=json.loads((DETAILS/'statistical_generation_manifest.json').read_text())
    if generation_manifest['decode_repeats']!=K or generation_manifest['benchmark_hash']!=plan['data_sha256']['benchmark.csv']:
        raise SystemExit('Statistical generation controls use different data or repetition counts. Rerun 02_fit_controls.py.')
    if generation_manifest['statistical_prediction_hash']!=hashlib.sha256((DETAILS/'statistical_predictions.csv').read_bytes()).hexdigest():
        raise SystemExit('Statistical predictions changed since generation controls were made. Rerun 02_fit_controls.py.')
    sampled_stats=pd.read_csv(DETAILS/'statistical_sampled_predictions.csv').set_index('profile_id').reindex(pop.index)
    if sampled_stats.isna().any().any():raise SystemExit('Statistical generation control predictions are incomplete.')
    failures=calls.groupby(['mode','role']).status.agg(total='size',valid=lambda s:int((s=='ok').sum())).reset_index()
    failures['failure_rate']=1-failures.valid/failures.total;failures.to_csv(OUT/'api_failure_rates.csv',index=False)
    summaries=[];groups=[];generation_sizes=[];paired=[];prediction_quality=[];associations=[];categorical=[];calibration=[]
    structure=[];components=[];probability_calibration=[]
    raw=pd.read_csv(DATA/'sampled_gss.csv').set_index('profile_id').reindex(pop.index)
    for role,part in calls.groupby('role'):
        cond=part[part['mode']=='conditional'];ok=cond[cond.status=='ok']
        counts=ok.groupby('profile_id').size()
        complete=(set(counts.index)==set(pop.index) and (counts==K).all())
        if not complete:
            summaries.append({'role':role,'status':'incomplete_outputs_inference_not_computed'});continue
        f=ok.groupby('profile_id')[TARGETS].mean().reindex(pop.index)
        first=ok.sort_values('decode').drop_duplicates('profile_id').set_index('profile_id').reindex(pop.index)
        first['group']=first.gender+'_'+np.where(first.age>=60,'60plus','under60')
        joint=part[(part['mode']=='joint')&(part.status=='ok')].copy()
        joint['group']=joint.gender+'_'+np.where(joint.age>=60,'60plus','under60')
        structure.extend(structure_descriptions(pop,first,joint,role))
        # These are descriptive relationship estimates, not causal effects.
        associations.append({'role':role,'statistic':'binary_target_phi',
                             'real':pop[TARGETS].corr().iloc[0,1],
                             'conditional':first[TARGETS].corr().iloc[0,1],
                             'joint':joint[TARGETS].corr().iloc[0,1] if len(joint)>1 else np.nan})
        for category in ['Very happy','Pretty happy','Not too happy']:
            categorical.append({'role':role,'variable':'happy','category':category,
                                'real_frequency':float((raw.happy==category).mean()),
                                'conditional_frequency':float((first.happy_category==category).mean()),
                                'joint_frequency':float((joint.happy_category==category).mean()) if len(joint) else np.nan})
        for target in TARGETS:
            y=pop[target].to_numpy(float);pred=f[target].to_numpy(float);gen=first[target].to_numpy(float)
            p=pop.groupby('group').size()/len(pop);mu=pop.groupby('group')[target].mean()
            row={'role':role,'target':target,'status':'computed','real_mean':y.mean(),'conditional_mean':gen.mean(),
                 'conditional_bias':gen.mean()-y.mean(),'brier':brier_score_loss(y,pred),
                 'auc':roc_auc_score(y,pred),'answer_variance_ratio':np.var(gen,ddof=1)/np.var(y,ddof=1),
                 'binary_js_similarity':1-jensenshannon([y.mean(),1-y.mean()],[gen.mean(),1-gen.mean()],base=2)**2,
                 'conditional_structure_bias':0.0}
            row.update(conditional_repeat_average=float(pred.mean()),conditional_repeat_average_bias=float(pred.mean()-y.mean()),
                       distribution_comparison_decode=0,decode_repeats=K)
            predictors={'大模型':pred,'logistic':stats['logistic_'+target].to_numpy(float),'forest':stats['forest_'+target].to_numpy(float)}
            if 'cart_'+target in stats.columns:predictors['cart']=stats['cart_'+target].to_numpy(float)
            for m in ['logistic','forest','cart']:
                predictors[m+'_generated']=sampled_stats[m+'_generated_'+target].to_numpy(float)
            paired.extend(paired_inference(pop,predictors,target,role,args.R))
            prediction_rows=quality_controls(pop,predictors,target,role)
            for entry in prediction_rows:
                bins=probability_bins(y,predictors[entry['model']],target,role,entry['model'])
                probability_calibration.extend(bins)
                entry['ece_10_fixed_bins']=float(sum(b['N']/len(y)*abs(b['mean_prediction']-b['observed_proportion']) for b in bins if b['N']))
            prediction_quality.extend(prediction_rows)
            descriptions=group_descriptions(pop,gen,target,role)
            for entry in descriptions:
                label=entry['group']
                if label in ['Female','Male']:mask=pop.gender.to_numpy()==label
                elif label=='under60':mask=pop.age.to_numpy()<60
                elif label=='60plus':mask=pop.age.to_numpy()>=60
                else:mask=pop.group.to_numpy()==label
                entry.update(prediction_group_mean=float(pred[mask].mean()),prediction_group_bias=float(pred[mask].mean()-y[mask].mean()),distribution_comparison_decode=0)
            groups.extend(descriptions)
            if len(joint):
                q=(joint.groupby('group').size()/len(joint)).reindex(p.index,fill_value=0)
                nu=joint.groupby('group')[target].mean().reindex(p.index)
                # Any empty group prevents joint-standardised estimation.
                observed=q>0
                BS=float(((q-p)*mu).sum())
                BR=float((q[observed]*(nu[observed]-mu[observed])).sum())
                for group in p.index:
                    has_group=bool(q[group]>0)
                    components.append(dict(role=role,target=target,group=group,p=float(p[group]),q=float(q[group]),
                                           real_mean=float(mu[group]),generated_mean=float(nu[group]) if has_group else np.nan,
                                           structure_contribution=float((q[group]-p[group])*mu[group]),
                                           answer_contribution=float(q[group]*(nu[group]-mu[group])) if has_group else 0.0,
                                           joint_calibrated_answer_contribution=float(p[group]*(nu[group]-mu[group])) if has_group else np.nan,
                                           group_observed=has_group))
                row.update(joint_records=len(joint),joint_mean=joint[target].mean(),structure_bias=BS,answer_bias=BR,
                           decomposition_error=joint[target].mean()-y.mean()-BS-BR,
                           joint_standardised_mean=float(p@nu) if nu.notna().all() else np.nan)
                row['joint_restricted_to_valid_outputs']=len(joint)<plan['joint_records']
                calibration.extend(calibration_descriptions(pop,joint,target,role))
                assert abs(row['decomposition_error'])<1e-10
            ixF=pop.gender.to_numpy()=='Female';ixM=~ixF
            row['real_sex_gap']=y[ixF].mean()-y[ixM].mean();row['generated_sex_gap']=gen[ixF].mean()-gen[ixM].mean()
            row['sex_gap_bias']=row['generated_sex_gap']-row['real_sex_gap']
            summaries.append(row)
            all_joint=part[part['mode']=='joint'].sort_values('profile_id')
            for m in [100,200,500]:
                subset=all_joint[all_joint.profile_id<m]
                entry={'role':role,'target':target,'m':m,'status':'incomplete_or_invalid_generation'}
                if len(subset)==m and (subset.status=='ok').all():
                    z=subset.copy();z['group']=z.gender+'_'+np.where(z.age>=60,'60plus','under60')
                    nu=z.groupby('group')[target].mean().reindex(p.index)
                    gm=z.groupby('gender')[target].mean()
                    entry.update(status='descriptive_single_generation_batch',raw_mean=z[target].mean(),
                                 raw_bias=z[target].mean()-y.mean(),
                                 standardised_mean=float(p@nu) if nu.notna().all() else np.nan,
                                 female_minus_male=gm.get('Female',np.nan)-gm.get('Male',np.nan))
                generation_sizes.append(entry)
    files={'api_quality':summaries,'api_group_means':groups,'api_inference':paired,
           'api_generation_sizes':generation_sizes,'api_paired_comparison':paired,
           'api_prediction_controls':prediction_quality,'api_associations':associations,
           'api_category_frequencies':categorical,'api_calibration':calibration,
           'api_structure_balance':structure,'api_bias_components':components,
           'api_probability_calibration':probability_calibration}
    for name,rows in files.items():
        if rows:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    success=len([r for r in summaries if r.get('status')=='computed'])==len(plan['roles'])*len(TARGETS)
    report={'conditional_inference_complete':success,'all_outputs_valid':bool((calls.status=='ok').all()),
            'frozen_predictions':True,'R':args.R,'reported_model':str(versions[0]),
            'joint_calibration_is_descriptive':True,'conditional_complete_roles':sorted(set(r['role'] for r in summaries if r.get('status')=='computed'))}
    (OUT/'evaluation_status.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not success:raise SystemExit('Some profiles have invalid/missing API outputs; inferential results were withheld for those conditions. Inspect api_failure_rates.csv.')
    print('大模型回答数据分析完成。')


def calibration_descriptions(pop,joint,target,role):
    """One generation batch: compare raw, marginal-raked and joint-calibrated means.

    These point estimates are not repeated-generation coverage experiments.
    """
    levels=sorted(pop.group.unique())
    p=pop.groupby('group').size().reindex(levels).to_numpy(float)/len(pop)
    q=joint.groupby('group').size().reindex(levels,fill_value=0).to_numpy(float)/len(joint)
    nu=joint.groupby('group')[target].mean().reindex(levels).to_numpy(float)
    truth=pop[target].mean()
    rows=[dict(role=role,target=target,method='直接联合生成',estimate=joint[target].mean(),
               bias=joint[target].mean()-truth,valid_records=len(joint),status='descriptive_single_batch')]
    if (q==0).any():
        rows.append(dict(role=role,target=target,method='联合分组校准',status='missing_joint_group_not_estimable'))
        return rows
    gender=np.array([s.split('_')[0] for s in levels]); age=np.array([s.split('_')[1] for s in levels])
    r=q.copy()
    for _ in range(10000):
        old=r.copy()
        for factor in [gender,age]:
            for value in np.unique(factor):
                ix=factor==value;r[ix]*=p[ix].sum()/r[ix].sum()
        if np.max(abs(r-old))<1e-12:break
    else:raise ValueError('Marginal calibration did not converge')
    for label,weights in [('性别与年龄边际校准',r),('联合分组校准',p)]:
        estimate=float(weights@nu)
        record_weights=weights/q/len(joint)
        eff=1/float(np.sum(len(joint)*q*record_weights**2))
        rows.append(dict(role=role,target=target,method=label,estimate=estimate,bias=estimate-truth,
                         effective_generated_n=eff,valid_records=len(joint),status='descriptive_single_batch'))
    return rows


def probability_bins(y,pred,target,role,model):
    pred=np.asarray(pred,float);index=np.minimum((pred*10).astype(int),9);rows=[]
    for i in range(10):
        ix=index==i;n=int(ix.sum())
        rows.append(dict(role=role,target=target,model=model,bin=i,lower=i/10,upper=(i+1)/10,N=n,
                         mean_prediction=float(pred[ix].mean()) if n else np.nan,
                         observed_proportion=float(y[ix].mean()) if n else np.nan))
    return rows


def structure_descriptions(pop,conditional,joint,role):
    rows=[]
    for variable in ['gender','education','marital_status','laborforce','age_band','gender_age_joint']:
        def series(frame):
            if variable=='age_band':
                return pd.cut(frame.age,[17,29,44,59,74,89],labels=['18-29','30-44','45-59','60-74','75plus'])
            if variable=='gender_age_joint':return frame['group']
            return frame[variable]
        real=series(pop).value_counts(normalize=True).sort_index()
        fixed=series(conditional).value_counts(normalize=True)
        generated=series(joint).value_counts(normalize=True) if len(joint) else pd.Series(dtype=float)
        for value in real.index:
            p=float(real[value]);q=float(generated.get(value,0)) if len(joint) else np.nan
            rows.append(dict(role=role,variable=variable,category=str(value),real_frequency=p,
                             conditional_frequency=float(fixed.get(value,0)),joint_frequency=q,
                             joint_frequency_difference=q-p,
                             generated_support_missing=bool(p>0 and q==0) if len(joint) else True))
    return rows

if __name__=='__main__':main()
