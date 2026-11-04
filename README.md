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
| [`paper_results/`](paper_results/) | 论文表 1–14、图 1–5 和正文引用数值 |
| [`paper_results/details/`](paper_results/details/) | 完整精度的中间结果，包括偏差分解、预测评价和推断对照 |

## 离线复现

在仓库根目录使用 Python 3.13。安装依赖后，脚本读取已保存的模型回答，重新计算论文图表。计算过程无需调用模型 API 或提供密钥。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_survey.txt
python run_paper.py
```

Windows 用户可用 `py -3.13 -m venv .venv` 创建环境，用 `.\.venv\Scripts\python.exe -m pip install -r requirements_survey.txt` 安装依赖，再运行 [`运行论文计算.bat`](运行论文计算.bat)。重算会更新 `paper_results/` 中的同名结果。可通过 [表格索引](paper_results/table_index.json) 和 [图形索引](paper_results/figure_index.json) 对照论文编号。

在 macOS 与 Python 3.13 上完成离线复算时，表 10 和表 12 的个别随机森林指标出现末位差异。仓库保留论文随附的原始图表。

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
