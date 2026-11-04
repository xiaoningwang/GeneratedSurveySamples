from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.patches import Rectangle,FancyArrowPatch
import pandas as pd

from paths import ROOT,DETAILS
OUT=DETAILS
font=FontProperties(fname=str(ROOT/'survey_code/fonts/NotoSerifCJKsc-Regular.otf'))
fontManager.addfont(font.get_file())
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':9})

g=pd.read_csv(OUT/'group_gap_theory.csv')
fig,ax=plt.subplots(figsize=(6.3,2.6));x=range(4)
ax.axhline(0,color='gray',lw=.6)
ax.plot(x,g.true_gap,'ko-',label='真实群体差值');ax.plot(x,g.generated_gap,'ks--',label='生成群体差值')
ax.set_xticks(list(x),g.population);ax.set_ylabel('群体四减群体一的均值差')
ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'group_gap.png',dpi=300);plt.close(fig)

s=pd.read_csv(OUT/'sample_sizes_mc.csv')
fig,axs=plt.subplots(1,2,figsize=(6.7,2.7))
for label,ls in [('联合分组校准','s--'),('残差校正','o-')]:
    v=s[(s.n==200)&(s.method==label)].sort_values('m')
    axs[0].plot(v.m,v.width,'k'+ls,label=label)
    axs[1].plot(v.m,v.coverage*100,'k'+ls,label=label)
axs[0].set_ylabel('平均区间宽度');axs[1].set_ylabel('覆盖率（%）')
for ax in axs:ax.set_xlabel('生成样本量 m');ax.set_xticks([500,2000,5000]);ax.legend(frameon=False,fontsize=7)
axs[1].axhline(95,color='gray',lw=.7,ls=':');fig.tight_layout();fig.savefig(OUT/'sample_sizes.png',dpi=300);plt.close(fig)

fig,ax=plt.subplots(figsize=(6.1,3.7));ax.set(xlim=(0,12),ylim=(0,7.3));ax.axis('off')
def box(x,y,w,h,text):
    ax.add_patch(Rectangle((x,y),w,h,fill=False,lw=.7,color='black'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8,linespacing=1.45)
def arr(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.7,color='black'))
box(2.3,6.35,7.4,.7,'固定真实调查基准、题项及群体差值')
box(.1,4.75,5.6,1.05,'同一真实协变量：仅生成回答\n隔离条件回答偏差')
box(6.3,4.75,5.6,1.05,'模型生成协变量及回答\n分解样本结构偏差与条件回答偏差')
arr((4,6.35),(2.9,5.8));arr((8,6.35),(9.1,5.8))
box(.1,3.1,5.6,1.0,'独立训练样本与相同真实标签预算\n逻辑回归、随机森林、DeepSeek')
box(6.3,3.1,5.6,1.0,'保留原始API响应、无效API输出及版本\n先小规模检验API连通，再完整调用')
arr((2.9,4.75),(2.9,4.1));arr((9.1,4.75),(9.1,4.1))
box(2.3,1.55,7.4,1.0,'分别检验总体均值、群体差值及区间覆盖\n真实样本量与生成样本量分别变化')
arr((2.9,3.1),(4,2.55));arr((9.1,3.1),(8,2.55))
box(2.3,.15,7.4,.8,'在预定总体与群体上检验统计用途')
arr((6,1.55),(6,.95));fig.tight_layout();fig.savefig(OUT/'api_workflow.png',dpi=300);plt.close(fig)

print('模拟与流程图已生成。')
