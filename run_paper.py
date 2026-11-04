"""Calculate the paper's tables and figures from the supplied input data."""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'survey_code'))

def main():
    from paths import create_folders,OUT,R
    create_folders()
    print('1/6 整理调查数据与固定样本划分',flush=True)
    from prepare_and_analyse import prepare,mechanisms
    prepare()
    print('2/6 拟合统计模型并计算统计对照',flush=True)
    from fit_models import run
    run()
    print('3/6 计算四种总体、群体差值及样本量模拟',flush=True)
    runpy.run_path(str(ROOT/'survey_code/simulate_means.py'),run_name='__main__')
    mechanisms(R)
    print('4/6 分析论文使用的大模型回答数据',flush=True)
    import evaluate_api
    saved=sys.argv[:]
    try:
        sys.argv=['evaluate_api.py','--R',str(R)];evaluate_api.main()
    finally:sys.argv=saved
    print('5/6 绘制论文图形',flush=True)
    runpy.run_path(str(ROOT/'survey_code/plot_mechanism.py'),run_name='__main__')
    runpy.run_path(str(ROOT/'survey_code/plot_details.py'),run_name='__main__')
    from export_results import export_tables,export_figures
    export_tables();export_figures()
    print('6/6 保存正文数值与结果说明',flush=True)
    from body_results import save
    save()
    (OUT/'结果说明.txt').write_text('table_01.csv—table_14.csv：论文表1—表14。\nfigure_01.png—figure_05.png：论文图1—图5。\n正文数值.csv：正文引用的主要计算数值。\ndetails文件夹：完整精度的中间数据与全部模型结果。\n表1、表3、表9为方法说明表；表2为文献输入及百分点差。\n分布比较用首次回答，推断用三次回答平均。\n',encoding='utf-8-sig')
    print('\n计算完成，全部结果保存在：'+str(OUT),flush=True)

if __name__=='__main__':main()
