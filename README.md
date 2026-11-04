# GeneratedSurveySamples

《大语言模型生成调查样本的统计偏差与推断有效性》的配套复现材料。论文作者为王小宁、侯祎龙、金勇进。


## 研究内容

研究区分两种生成方式。一种在给定真实协变量的条件下生成调查回答，另一种联合生成协变量和回答。论文用总体比例、群体差值和置信区间覆盖率，分别考察生成样本的统计保真度和推断有效性。实验包括机制模拟、基于 GSS 2018 子样本的模型生成，以及在相同输入信息或真实标签预算下的统计方法对照。

机制模拟表明，样本结构偏差与条件回答偏差可以相互抵消，使总体比例接近基准，同时保留群体差值的失真。扩大生成样本量会降低生成随机误差，系统性偏差仍会保留。在本研究的 DeepSeek 实验中，真实观测的残差校正使均值偏差接近零。少样本提示下的校正估计在相同真实标签预算下未取得精度优势。

## 材料索引

| 路径 | 内容 |
| --- | --- |
| [`survey_code/`](survey_code/) | 数据整理、统计模型、机制模拟、模型回答评估及图表导出 |
| [`survey_data/`](survey_data/) | 固定样本、已保存的模型回答、实验元数据及来源说明 |
| [`survey_data/api_raw/`](survey_data/api_raw/) | 原始正式响应8,708条、试运行32条、请求计划与结束日志 |
| [`docs/`](docs/) | 数据来源、变量字典、随机过程、图表映射、公式推导与核对记录 |
| [`paper_results/`](paper_results/) | 论文表 1–14、图 1–5 和正文引用数值 |
| [`paper_results/details/`](paper_results/details/) | 完整精度的中间结果，包括偏差分解、预测评价和推断对照 |

## 离线复现

在仓库根目录使用 Python 3.13。安装依赖后，脚本读取已保存的模型回答，重新计算论文图表。计算过程无需调用模型 API 或提供密钥。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_survey.txt
python run_paper.py --output-dir reproduction_results --verify
```

Windows PowerShell可直接运行（项目可放在 `D:\Python-hyl\GeneratedSurveySamples`）：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_survey.txt
.\.venv\Scripts\python.exe run_paper.py --output-dir reproduction_results --verify
```

也可双击 [`运行论文计算.bat`](运行论文计算.bat)。全部重算结果保存在 `reproduction_results/`；原随稿结果 `paper_results/` 保留。原有 `python run_paper.py` 命令仍将结果写入 `paper_results/`。

程序依次校验原始输入、逐条重建并解析原始API记录、整理数据、拟合统计模型、运行模拟、评价生成回答并导出图表。结果目录的 `logs/run.log`、`environment.json`、`run_status.json`、`input_audit.json`、`api_replay.json` 和 `verification.json` 保存过程与核对结果。核对不通过时返回非零退出码，并列出具体差异。

2026年11月4日在 Linux、Python 3.13.5及指定五个计算包版本下，完整复算通过：14张表按显示精度一致，260条正文数值通过核对；原始响应重解析与分析数据一致。原Windows环境与此次验证环境分别存档。仓库原README记录过macOS下随机森林指标的末位差异，跨系统重算以实际核对报告为准。

## 期刊资料说明

- [数据来源与完整处理步骤](docs/data_provenance.md)
- [变量字典](docs/variable_dictionary.csv)
- [随机过程、种子、软件和操作系统](docs/randomness.md)
- [表1—14、图1—5及正文数值的程序映射](docs/result_map.md)
- [公式推导与方差计算口径](docs/derivations.md)
- [核对结果及尚需作者确认的信息](docs/journal_check.md)

首次下载公开数据的确切日期未写入原始清单，须由作者依据实际下载记录补充。当前材料明确保留了这一信息缺口，没有推定下载日期。

运行回归保护检查：

```bash
python -m unittest discover -s tests -v
```

## API 采集入口

[`run_api_pilot.py`](run_api_pilot.py) 和 [`run_api_formal.py`](run_api_formal.py) 分别对应试运行和正式采集。使用 `--dry-run` 可检查请求队列，无需提供密钥或发出请求。

```bash
python run_api_pilot.py --dry-run
python run_api_formal.py --dry-run
```

实际采集会交互式请求密钥，并将结果保存到 `paper_results/llm_collection/`。重新采集的回答可能与论文使用的缓存不同。核对论文结果时使用离线复现入口。

## 数据来源与推断范围

GSS 2018 数据来自 Xie 等（2026）公开的 1,000 条处理后子样本。论文保留协变量及两个目标题项均完整的 659 条记录，按固定随机种子划分为 100 条训练记录和 559 条评价记录。估计对象为这 559 条评价记录构成的有限总体，分析两个目标题项的总体比例和群体差值。 [上游数据与处理说明](https://github.com/LemengLiang/SSDataBench-Data) · [Xie 等（2026）](https://doi.org/10.1073/pnas.2538145123)

引用本研究时，可使用论文标题及作者王小宁、侯祎龙、金勇进，并注明所用稿件版本。使用上游调查数据时，请一并注明其来源。
