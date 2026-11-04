# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np

from paths import MECHANISM
OUT=MECHANISM
OUT.mkdir(parents=True,exist_ok=True)
R, n, m = 2000, 200, 5000
seed = 20261010
rng = np.random.default_rng(seed)
cells = np.array([[0,0],[0,1],[1,0],[1,1]])
scenarios = [
    ('总体一', [.25,.25,.25,.25], [.50,.20,.20,.10], [.10,.35,.55,.80], [.08,.08,.08,.08]),
    ('总体二', [.40,.10,.10,.40], [.25,.25,.25,.25], [.10,.20,.30,.90], [.05,.10,-.05,.08]),
    ('总体三', [.70,.20,.08,.02], [.50,.25,.20,.05], [.10,.30,.50,.90], [.02,.03,.05,-.20]),
    ('总体四', [.25,.25,.25,.25], [.50,.20,.20,.10], [.10,.35,.55,.80], [.70,.20,-.20,-.70]),
]

def rake(q, p):
    t = q.copy()
    for _ in range(1000):
        prev=t.copy()
        for j in range(2):
            for a in (0,1):
                z=cells[:,j]==a
                t[z]*=p[z].sum()/t[z].sum()
        if np.max(abs(t-prev))<1e-13:
            return t
    raise RuntimeError('raking did not converge')

results=[]
repetition_arrays={}
for name,p,q,mu,b in scenarios:
    p,q,mu,b=map(lambda x:np.array(x,float),(p,q,mu,b))
    f=mu+b
    assert np.all((f>0)&(f<1)) and np.isclose(p.sum(),1) and np.isclose(q.sum(),1)
    truth=float(p@mu)
    mh=(m*q).astype(int)
    assert mh.sum()==m and np.all(mh>1)
    # Independent population draws for each repetition; f is fixed before sampling.
    h=rng.choice(4,(R,n),p=p)
    y=rng.binomial(1,mu[h])
    resid=y-f[h]
    gh=rng.binomial(mh[None,:],f[None,:],size=(R,4))/mh
    phat=rake(q,p)
    estimates={}
    estimates['真实样本均值']=(y.mean(1),np.sqrt(y.var(1,ddof=1)/n))
    for label,t in [('直接生成均值',q),('边际校准',phat),('联合分组校准',p)]:
        est=gh@t
        # Fixed group quotas. The unbiased within-cell variance estimate is
        # mh/(mh-1) * gh*(1-gh); each group mean's variance divides it by mh.
        se=np.sqrt(np.sum(t*t*gh*(1-gh)/(mh-1),axis=1))
        estimates[label]=(est,se)
    estimates['残差校正 正确分布']=(p@f+resid.mean(1),np.sqrt(resid.var(1,ddof=1)/n))
    estimates['残差校正 错误分布']=(q@f+resid.mean(1),np.sqrt(resid.var(1,ddof=1)/n))
    rec={'name':name,'p':p.tolist(),'q':q.tolist(),'mu':mu.tolist(),'b':b.tolist(),'f':f.tolist(),'truth':truth,'raked_p':phat.tolist(),'methods':[]}
    for method_index,(label,(est,se)) in enumerate(estimates.items()):
        repetition_arrays[f's{len(results)+1}_m{method_index+1}_estimate']=est
        repetition_arrays[f's{len(results)+1}_m{method_index+1}_se']=se
        coverage=np.mean(abs(est-truth)<=1.96*se)
        stat={'method':label,'bias':float(np.mean(est-truth)),'bias_mcse':float(np.std(est,ddof=1)/np.sqrt(R)),'rmse':float(np.sqrt(np.mean((est-truth)**2))), 'coverage':float(coverage),'coverage_mcse':float(np.sqrt(coverage*(1-coverage)/R)), 'width':float(np.mean(3.92*se)),'variance':float(np.var(est,ddof=1))}
        if label=='真实样本均值': theory_bias=0
        elif label=='残差校正 正确分布': theory_bias=0
        elif label=='残差校正 错误分布': theory_bias=float((q-p)@f)
        else:
            t={'直接生成均值':q,'边际校准':phat,'联合分组校准':p}[label]
            theory_bias=float(t@f-truth)
        stat['theory_bias']=theory_bias
        assert abs(stat['bias']-theory_bias) < 5*stat['bias_mcse']+1e-12
        rec['methods'].append(stat)
    hv=rec['methods'][0]['variance']; pv=rec['methods'][4]['variance']
    rec['variance_reduction_pct']=100*(1-pv/hv)
    rec['composition_bias']=float((q-p)@mu)
    rec['conditional_bias']=float(q@b)
    results.append(rec)

payload={'R':R,'n':n,'m':m,'seed':seed,'rng':'numpy.default_rng PCG64','ci':'Wald interval estimate +/- 1.96 SE','scenarios':results,'numpy_version':np.__version__}
np.savez_compressed(OUT/'repetitions.npz',**repetition_arrays)
(OUT/'simulation_results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
for s in results:
    print(s['name'],f"总体均值={s['truth']:.4f}",f"估计方差降低比例={s['variance_reduction_pct']:.2f}%")
    for t in s['methods']:
        print(t['method'],f"bias={t['bias']:.4f} RMSE={t['rmse']:.4f} coverage={t['coverage']*100:.2f}% width={t['width']:.4f}")
