"""Reproducible real-data benchmarks and mechanism checks. No LLM results fabricated."""
from pathlib import Path
import argparse, json, hashlib, urllib.request, os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from paths import ROOT,DATA,DETAILS
OUT=DETAILS
SEED=20261012
FEATURES=['gender','age','education','marital_status','laborforce']
TARGETS=['agree','very_happy']
SOURCE='https://raw.githubusercontent.com/LemengLiang/SSDataBench-Data/main/processed_survey_data/sampled_gss.csv'
SOURCE_SHA256='7b86f23d24b3355eb2094ccee1a5693e8950d55fef6db875d3c4bdb82e1232dc'

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(DATA/'sampled_gss.csv')
    df=raw.dropna(subset=FEATURES+['gender_role_attitude','happy']).copy()
    df['agree']=(df.gender_role_attitude=='Agree').astype(int)
    df['very_happy']=(df.happy=='Very happy').astype(int)
    df['profile_id']=df.profile_id.astype(int)
    df['group']=df.gender+'_'+np.where(df.age>=60,'60plus','under60')
    df=df[['profile_id']+FEATURES+['group']+TARGETS].reset_index(drop=True)
    ix=np.random.default_rng(SEED).permutation(len(df))
    train=pd.read_csv(DATA/'training.csv');pop=pd.read_csv(DATA/'benchmark.csv')
    pd.testing.assert_frame_equal(train.reset_index(drop=True),df.iloc[ix[:100]].reset_index(drop=True),check_dtype=False)
    pd.testing.assert_frame_equal(pop.reset_index(drop=True),df.iloc[ix[100:]].reset_index(drop=True),check_dtype=False)
    df.to_csv(OUT/'complete_cases.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for label,block in [('全部',pop),('女性',pop[pop.gender=='Female']),('男性',pop[pop.gender=='Male']),('60岁以下',pop[pop.age<60]),('60岁及以上',pop[pop.age>=60])]:
        rows.append({'group':label,'N':len(block),'agree':block.agree.mean(),'very_happy':block.very_happy.mean()})
    pd.DataFrame(rows).to_csv(OUT/'real_benchmark.csv',index=False,encoding='utf-8-sig')
    return train,pop

def fit_controls(train,pop):
    cat=[c for c in FEATURES if c!='age']
    result=pop[['profile_id']].copy()
    for target in TARGETS:
        for name,model in [('logistic',LogisticRegression(max_iter=2000,C=1.0)),
                           ('forest',RandomForestClassifier(n_estimators=200,min_samples_leaf=5,max_depth=5,random_state=SEED,n_jobs=1)),
                           ('cart',DecisionTreeClassifier(max_depth=5,min_samples_leaf=5,random_state=SEED))]:
            pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat),('age',StandardScaler(),['age'])])
            pipe=make_pipeline(pre,model);pipe.fit(train[FEATURES],train[target])
            result[f'{name}_{target}']=pipe.predict_proba(pop[FEATURES])[:,list(pipe[-1].classes_).index(1)]
    result.to_csv(OUT/'statistical_predictions.csv',index=False)

def finite_population_check(pop,predictions,R=2000):
    # Independent SRS without replacement from the fixed empirical population.
    # Normal intervals; conditional on frozen training and predictions.
    rng=np.random.default_rng(SEED+1);N=len(pop);rows=[]
    ns=[50,100,200]
    for target in TARGETS:
        y=pop[target].to_numpy(float);truth=y.mean()
        for n in ns:
            ids=np.array([rng.choice(N,n,replace=False) for _ in range(R)])
            yy=y[ids]
            methods={'真实样本均值':(yy.mean(1),np.sqrt((1-n/N)*yy.var(1,ddof=1)/n))}
            for model in ['logistic','forest']:
                f=predictions[f'{model}_{target}'].to_numpy(float);e=y-f;ee=e[ids]
                methods[model+'残差校正']=(f.mean()+ee.mean(1),np.sqrt((1-n/N)*ee.var(1,ddof=1)/n))
            # Stronger same-label-budget control. Training labels are counted as cost.
            b=n+100
            ib=np.array([rng.choice(N,b,replace=False) for _ in range(R)])
            z=y[ib];methods['等标签预算直接估计']=(z.mean(1),np.sqrt((1-b/N)*z.var(1,ddof=1)/b))
            for label,(est,se) in methods.items():
                rows.append({'target':target,'n':n,'observed_label_budget':n if label=='真实样本均值' else n+100,'method':label,
                             'bias':np.mean(est-truth),'rmse':np.sqrt(np.mean((est-truth)**2)),
                             'coverage':np.mean(abs(est-truth)<=1.96*se),'width':np.mean(3.92*se),
                             'variance':np.var(est,ddof=1),'R':R,'conditional_on_fixed_predictions':True})
    pd.DataFrame(rows).to_csv(OUT/'real_controls.csv',index=False)

def mechanisms(R=2000):
    rng=np.random.default_rng(SEED+2)
    scenarios=[('总体一',[.25]*4,[.50,.20,.20,.10],[.10,.35,.55,.80],[.08]*4),
               ('总体二',[.40,.10,.10,.40],[.25]*4,[.10,.20,.30,.90],[.05,.10,-.05,.08]),
               ('总体三',[.70,.20,.08,.02],[.50,.25,.20,.05],[.10,.30,.50,.90],[.02,.03,.05,-.20]),
               ('总体四',[.25]*4,[.50,.20,.20,.10],[.10,.35,.55,.80],[.70,.20,-.20,-.70])]
    rows=[];theory=[]
    for name,p,q,mu,b in scenarios:
        p,q,mu,b=[np.array(z,float) for z in (p,q,mu,b)];f=mu+b
        h=rng.choice(4,(R,200),p=p);y=rng.binomial(1,mu[h]);e=y-f[h]
        gh=rng.binomial((5000*q).astype(int),f,size=(R,4))/(5000*q)
        gap=mu[3]-mu[0];syn_gap=f[3]-f[0]
        mask0=h==0;mask3=h==3;n0=mask0.sum(1);n3=mask3.sum(1);ok=(n0>=2)&(n3>=2)
        def group_summary(a,mask,n):
            mean=(a*mask).sum(1)/n
            var=((a*a*mask).sum(1)-n*mean**2)/(n-1)
            return mean,var
        # Preserve failed subgroup repeats, rather than silently removing them.
        with np.errstate(divide='ignore',invalid='ignore'):
            m0,v0=group_summary(y,mask0,n0);m3,v3=group_summary(y,mask3,n3)
            real=m3-m0;rs=np.sqrt(v3/n3+v0/n0)
            e0,ev0=group_summary(e,mask0,n0);e3,ev3=group_summary(e,mask3,n3)
            pp=f[3]+e3-f[0]-e0;ps=np.sqrt(ev3/n3+ev0/n0)
        syn=gh[:,3]-gh[:,0]
        ss=np.sqrt(gh[:,3]*(1-gh[:,3])/(5000*q[3]-1)+gh[:,0]*(1-gh[:,0])/(5000*q[0]-1))
        for label,est,se,valid in [('真实群体差值',real,rs,ok),('合成群体差值',syn,ss,np.ones(R,bool)),('群体残差校正',pp,ps,ok)]:
            vals=est[valid];errs=vals-gap
            cp=np.sum(valid&(abs(est-gap)<=1.96*se))/R # failures count as noncoverage
            rows.append({'population':name,'method':label,'true_gap':gap,'bias_on_valid':errs.mean(),
                         'rmse_on_valid':np.sqrt(np.mean(errs**2)),'coverage_all_repeats':cp,
                         'width_on_valid':np.mean(3.92*se[valid]),'failure_rate':1-valid.mean(),'R':R})
        theory.append({'population':name,'mean_bias_joint':float(p@b),'true_gap':float(gap),
                       'generated_gap':float(syn_gap),'gap_bias':float(b[3]-b[0]),
                       'group_a_bias':float(b[3]),'group_b_bias':float(b[0])})
    pd.DataFrame(rows).to_csv(OUT/'group_gap_mc.csv',index=False)
    pd.DataFrame(theory).to_csv(OUT/'group_gap_theory.csv',index=False)
    # Sample size varies separately for genuine observations and generated answers.
    p=np.array([.25]*4);q=np.array([.5,.2,.2,.1]);mu=np.array([.1,.35,.55,.8]);f=mu+.08;truth=p@mu
    h=rng.choice(4,(R,500),p=p);y=rng.binomial(1,mu[h]);sizes=[]
    for m in [500,2000,5000]:
        mh=(m*q).astype(int);gh=rng.binomial(mh,f,size=(R,4))/mh
        joint=gh@p;js=np.sqrt(np.sum(p*p*gh*(1-gh)/(mh-1),axis=1))
        for n in [50,200,500]:
            z=y[:,:n];rr=z-f[h[:,:n]]
            for label,est,se in [('真实样本均值',z.mean(1),np.sqrt(z.var(1,ddof=1)/n)),
                                 ('联合分组校准',joint,js),
                                 ('残差校正',p@f+rr.mean(1),np.sqrt(rr.var(1,ddof=1)/n))]:
                sizes.append({'n':n,'m':m,'method':label,'bias':np.mean(est-truth),
                              'rmse':np.sqrt(np.mean((est-truth)**2)),'coverage':np.mean(abs(est-truth)<=1.96*se),
                              'width':np.mean(3.92*se),'R':R})
    pd.DataFrame(sizes).to_csv(OUT/'sample_sizes_mc.csv',index=False)

