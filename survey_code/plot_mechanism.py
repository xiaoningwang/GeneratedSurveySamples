# -*- coding: utf-8 -*-
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT=Path(__file__).parent.resolve()
from paths import MECHANISM
OUT=MECHANISM
OUT.mkdir(parents=True,exist_ok=True)
stats=json.loads((OUT/'simulation_results.json').read_text(encoding='utf-8'))
font=FontProperties(fname=str(ROOT/'fonts/NotoSerifCJKsc-Regular.otf'))
fontManager.addfont(font.get_file())
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':7.5,'axes.linewidth':.5,'xtick.major.width':.5,'ytick.major.width':.5,'xtick.direction':'in','ytick.direction':'in'})

# Journal graphics: monochrome, closed axes, thin lines and inward ticks.
fig,ax=plt.subplots(figsize=(12/2.54,4.5/2.54))
labels=['真实样本均值','联合分组校准','残差校正 正确分布','残差校正 错误分布']
dash=['solid',(0,(5,2)),(0,(1,1)),(0,(5,1,1,1))]
for k,(label,ls) in enumerate(zip(labels,dash)):
    vals=[next(t['coverage']*100 for t in s['methods'] if t['method']==label) for s in stats['scenarios']]
    ax.plot(range(1,5),vals,color='black',lw=.8,ls=ls,label=label.replace('真实','真实').replace(' ','（')+('）' if ' ' in label else ''))
ax.axhline(95,color='#777777',lw=.5,ls=(0,(2,3)))
ax.text(2.8,85,'名义95%',fontproperties=font,fontsize=6.5)
ax.set_xticks(range(1,5),['总体一','总体二','总体三','总体四'],fontproperties=font)
ax.set_yticks([0,25,50,75,100]);ax.set_ylim(-3,105);ax.set_xlim(.9,4.1)
ax.set_ylabel('覆盖率（%）',fontproperties=font,fontsize=7,labelpad=2)
ax.legend(loc='center left',bbox_to_anchor=(.08,.5),frameon=False,prop=FontProperties(fname=font.get_file(),size=6.8),handlelength=3)
fig.subplots_adjust(left=.115,right=.94,bottom=.19,top=.97)
fig.savefig(OUT/'figure1_coverage.png',dpi=400)
fig.savefig(OUT/'figure1_coverage.pdf')
plt.close(fig)
