"""Calculate paper results with input audits, a complete log and regression checks."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
import argparse
import os
import runpy
import sys
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'survey_code'))


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def calculate(verify):
    from paths import create_folders, OUT, R
    from reproducibility import check_inputs, audit_original_api, environment, write_json, verify_outputs
    create_folders()
    write_json(OUT/'logs/environment.json', environment())
    write_json(OUT/'logs/input_audit.json', check_inputs())
    audit_original_api(OUT)
    print('原始输入和8740条API响应核对通过；没有调用API。', flush=True)
    print('1/6 整理调查数据与固定样本划分', flush=True)
    from prepare_and_analyse import prepare, mechanisms
    prepare()
    print('2/6 拟合统计模型并计算统计对照', flush=True)
    from fit_models import run
    run()
    print('3/6 计算四种总体、群体差值及样本量模拟', flush=True)
    runpy.run_path(str(ROOT/'survey_code/simulate_means.py'), run_name='__main__')
    mechanisms(R)
    print('4/6 分析论文使用的大模型回答数据', flush=True)
    import evaluate_api
    saved = sys.argv[:]
    try:
        sys.argv = ['evaluate_api.py', '--R', str(R)]
        evaluate_api.main()
    finally:
        sys.argv = saved
    print('5/6 绘制论文图形', flush=True)
    runpy.run_path(str(ROOT/'survey_code/plot_mechanism.py'), run_name='__main__')
    runpy.run_path(str(ROOT/'survey_code/plot_details.py'), run_name='__main__')
    from export_results import export_tables, export_figures
    export_tables()
    export_figures()
    print('6/6 保存正文数值与结果说明', flush=True)
    from body_results import save
    save()
    (OUT/'结果说明.txt').write_text(
        'table_01.csv—table_14.csv：论文表1—表14。\nfigure_01.png—figure_05.png：论文图1—图5。\n'
        '正文数值.csv：正文引用的主要计算数值。\ndetails：完整精度的中间数据与全部模型结果。\n'
        'logs：本次运行日志、环境、输入校验及API重解析报告。\n'
        '表1、表3、表9为方法说明表；表2为文献输入及百分点差。\n'
        '分布比较用首次回答，推断用三次回答平均。\n', encoding='utf-8-sig')
    if verify:
        verify_outputs(OUT)
        print('核对通过：论文14张表、正文数值及5幅图文件。', flush=True)
    print('\n计算完成，全部结果保存在：' + str(OUT), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'paper_results',
                        help='结果目录；建议使用 reproduction_results 保留仓库基准。相对路径基于仓库根目录。')
    parser.add_argument('--verify', action='store_true', help='与冻结的论文表格和正文数值逐项比较。')
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
    output = output.resolve()
    if output == ROOT or output == ROOT/'survey_data' or ROOT/'survey_data' in output.parents:
        parser.error('结果目录不能覆盖代码根目录或原始数据目录。')
    os.environ['SURVEY_OUTPUT_DIR'] = str(output)
    (output/'logs').mkdir(parents=True, exist_ok=True)
    from reproducibility import write_json
    started = datetime.now(timezone.utc).isoformat()
    with (output/'logs/run.log').open('w', encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                print('Started UTC: ' + started, flush=True)
                calculate(args.verify)
            except BaseException:
                traceback.print_exc()
                write_json(output/'logs/run_status.json', {'status': 'failed', 'started_utc': started,
                           'ended_utc': datetime.now(timezone.utc).isoformat()})
                raise
            write_json(output/'logs/run_status.json', {'status': 'passed', 'started_utc': started,
                       'ended_utc': datetime.now(timezone.utc).isoformat(), 'verification_requested': args.verify})


if __name__ == '__main__':
    main()
