"""Paired finite-population comparisons conditional on frozen predictors.

This module does not generate or stand in for any API output.
"""
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import brier_score_loss, roc_auc_score


def interval_summary(est, se, truth, R, valid=None):
    est, se = np.asarray(est), np.asarray(se)
    if valid is None:
        valid = np.isfinite(est) & np.isfinite(se)
    covered = valid & (np.abs(est - truth) <= 1.96 * se)
    k = int(covered.sum())
    errors = est[valid] - truth
    return dict(
        bias=float(errors.mean()) if len(errors) else np.nan,
        rmse=float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan,
        coverage=k/R, coverage_count=k, R=R,
        coverage_mcse=float(np.sqrt(k/R*(1-k/R)/R)),
        coverage_lower_one_sided_95=float(beta.ppf(.05, k, R-k+1)) if k else 0.0,
        coverage_ci_lower=float(beta.ppf(.025, k, R-k+1)) if k else 0.0,
        coverage_ci_upper=float(beta.ppf(.975, k+1, R-k)) if k<R else 1.0,
        width=float(np.mean(3.92*se[valid])) if valid.any() else np.nan,
        variance=float(np.var(est[valid], ddof=1)) if valid.sum()>1 else np.nan,
        failure_rate=float(1-valid.mean()),
        frozen_predictions=True,
    )


def subgroup_mean_var(z, mask, count):
    with np.errstate(divide='ignore', invalid='ignore'):
        mean=(z*mask).sum(axis=1)/count
        var=((z*z*mask).sum(axis=1)-count*mean**2)/(count-1)
    # Floating point cancellation can produce a tiny negative variance.
    return mean, np.maximum(var, 0)


def paired_inference(pop, predictions, target, role, R=2000, seed=20261019):
    """All models see the same correction samples, with label budgets reported.

    predictions maps names to frozen probabilities in pop row order. Statistical
    controls and few-shot LLM predictions consume the independent 100-case set.
    """
    y=pop[target].to_numpy(float); N=len(y); truth=float(y.mean())
    rows=[]
    female=pop.gender.to_numpy()=='Female'
    older=pop.age.to_numpy()>=60
    contrasts={
        'female_minus_male':(female, ~female),
        '60plus_minus_under60':(older, ~older),
        'female_60plus_minus_male_under60':(female & older, ~female & ~older),
    }
    for n in [50,100,200]:
        if n>=N:
            raise ValueError('The correction sample must be smaller than the population.')
        rng=np.random.default_rng(seed+n)
        ids=np.array([rng.choice(N,n,replace=False) for _ in range(R)])
        z=y[ids]
        method_arrays={'真实样本均值':(z.mean(1),np.sqrt((1-n/N)*z.var(1,ddof=1)/n))}
        for name,f in predictions.items():
            f=np.asarray(f,float)
            if len(f)!=N or not np.isfinite(f).all() or ((f<0)|(f>1)).any():
                raise ValueError('Predictions must be finite probabilities aligned to the population.')
            residual=y-f; e=residual[ids]
            method_arrays[name+'残差校正']=(f.mean()+e.mean(1),np.sqrt((1-n/N)*e.var(1,ddof=1)/n))
        b=n+(100 if role=='fewshot' else 0)
        budget_ids=np.array([rng.choice(N,b,replace=False) for _ in range(R)])
        budget_y=y[budget_ids]
        method_arrays['大模型等标签预算直接估计']=(budget_y.mean(1),np.sqrt((1-b/N)*budget_y.var(1,ddof=1)/b))
        if b==n:
            method_arrays['大模型等标签预算直接估计']=method_arrays['真实样本均值']
        real_variance=np.var(method_arrays['真实样本均值'][0],ddof=1)
        budget_variance=np.var(method_arrays['大模型等标签预算直接估计'][0],ddof=1)
        for name,(est,se) in method_arrays.items():
            summary=interval_summary(est,se,truth,R)
            label_budget=n if name=='真实样本均值' else (b if name.startswith('大模型') else n+100)
            rows.append(dict(role=role,target=target,n=n,method=name,label_budget=label_budget,
                             **summary,
                             variance_reduction_vs_n=1-summary['variance']/real_variance,
                             variance_reduction_vs_llm_budget=1-summary['variance']/budget_variance,
                             paired_error_covariance_vs_n=float(np.cov(est,method_arrays['真实样本均值'][0],ddof=1)[0,1])))
        for suffix,(a,bmask) in contrasts.items():
            a_s=a[ids]; b_s=bmask[ids]; na=a_s.sum(1); nb=b_s.sum(1)
            valid=(na>=2)&(nb>=2)
            group_truth=float(y[a].mean()-y[bmask].mean())
            for name,f in [('真实群体差值',np.zeros(N))]+list(predictions.items()):
                f=np.asarray(f,float); e=(y-f)[ids]
                ma,va=subgroup_mean_var(e,a_s,na); mb,vb=subgroup_mean_var(e,b_s,nb)
                est=f[a].mean()+ma-f[bmask].mean()-mb
                se=np.sqrt((1-na/a.sum())*va/np.maximum(na,1)+(1-nb/bmask.sum())*vb/np.maximum(nb,1))
                method=name if name=='真实群体差值' else name+'群体残差校正'
                rows.append(dict(role=role,target=target+'_'+suffix,n=n,method=method,
                                 truth=group_truth,
                                 label_budget=n if name=='真实群体差值' else n+(100 if name!='大模型' or role=='fewshot' else 0),
                                 **interval_summary(est,se,group_truth,R,valid)))
    return rows


def quality_controls(pop, predictions, target, role):
    y=pop[target].to_numpy(float)
    result=[]
    for name,f in predictions.items():
        f=np.asarray(f,float)
        row=dict(role=role,target=target,model=name,brier=float(brier_score_loss(y,f)),
                 auc=float(roc_auc_score(y,f)),prediction_mean=float(f.mean()),
                 observed_mean=float(y.mean()),residual_variance=float(np.var(y-f,ddof=1)))
        result.append(row)
    return result


def group_descriptions(pop, generated, target, role):
    y=pop[target].to_numpy(float); z=np.asarray(generated,float)
    blocks={'Female':pop.gender.to_numpy()=='Female',
            'Male':pop.gender.to_numpy()=='Male',
            'under60':pop.age.to_numpy()<60,
            '60plus':pop.age.to_numpy()>=60}
    for group in sorted(pop.group.unique()):
        blocks[group]=pop.group.to_numpy()==group
    return [dict(role=role,target=target,group=group,N=int(ix.sum()),
                 real_mean=float(y[ix].mean()),generated_mean=float(z[ix].mean()),
                 bias=float(z[ix].mean()-y[ix].mean())) for group,ix in blocks.items()]
