# 论文结果与程序对应

对应《大语言模型生成调查样本的统计偏差与推断有效性》2026年11月3日三稿修改版。稿件指纹与冻结表格保存在 `survey_data/paper_reference.json`。所有输出集中在指定结果目录；默认 `paper_results/`，审查重算推荐 `reproduction_results/`。

| 论文结果 | 输出文件 | 计算入口与输入 |
| --- | --- | --- |
| 表1 信息来源及其统计作用 | `table_01.csv` | `concept_tables.py` 方法说明 |
| 表2 领域微调前后通过率 | `table_02.csv` | `export_results.calculate_tables`；`published_inputs.json`，文献原数值及百分点差 |
| 表3 评价对象、指标与判定依据 | `table_03.csv` | `concept_tables.py` 方法说明 |
| 表4 四种总体参数 | `table_04.csv` | `simulate_means.py` 中p、q、mu、b |
| 表5 六种方法模拟结果 | `table_05.csv` | `simulate_means.py` → `details/mechanism/simulation_results.json` |
| 表6 总体偏差与群体差值 | `table_06.csv` | `prepare_and_analyse.mechanisms` → `details/group_gap_theory.csv`，由设定参数精确计算 |
| 表7 样本量敏感性 | `table_07.csv` | 同上 → `details/sample_sizes_mc.csv` |
| 表8 评价总体基本特征 | `table_08.csv` | `prepare_and_analyse.prepare` → `details/real_benchmark.csv` |
| 表9 方法对照 | `table_09.csv` | `concept_tables.py` 实验设计说明 |
| 表10 统计模型推断对照 | `table_10.csv` | `fit_models.run`、`paired_evaluation.paired_inference` → `details/real_paired_controls.csv` |
| 表11 条件与联合生成结果 | `table_11.csv` | `evaluate_api.main` → `details/api/api_quality.csv` |
| 表12 预测质量 | `table_12.csv` | 同上 → `details/api/api_prediction_controls.csv`，预测取三次平均 |
| 表13 均值推断与标签预算 | `table_13.csv` | 同上 → `details/api/api_inference.csv`，n=200 |
| 表14 群体差值推断 | `table_14.csv` | 同上，fewshot、n=200、三个预设差值 |
| 图1 四总体覆盖率 | `figure_01.png` | `plot_mechanism.py`，`details/mechanism/simulation_results.json`；与表5对应 |
| 图2 群体差值方向 | `figure_02.png` | `plot_details.py`，`details/group_gap_theory.csv`；与表6对应 |
| 图3 生成样本量效应 | `figure_03.png` | `plot_details.py`，`details/sample_sizes_mc.csv`，n=200；与表7对应 |
| 图4 实施流程 | `figure_04.png` | `plot_details.py`，研究设计流程图，无随机数据 |
| 图5 真实样本量与推断精度 | `figure_05.png`、`.pdf` | `export_results.export_figures`，fewshot、agree、n=50/100/200，4种方法；绘图数据另存 `figure_05_data.csv` |
| 正文主要数值 | `正文数值.csv` | `body_results.save` 从模拟及API明细计算，共260条 |

## 正文中的明细位置

- 方差降低26.96%、56.39%、17.82%及方差增加82.01%：`details/mechanism/simulation_results.json` 的 `variance_reduction_pct`。
- 总体四差值偏差、总体三9.35%区间失败率及36.60%覆盖率：`details/group_gap_mc.csv`，失败重复没有从分母删除。
- 样本结构贡献与条件回答贡献：`details/api/api_bias_components.csv`；汇总在 `api_quality.csv` 的 `structure_bias`、`answer_bias`。
- 联合分组校准结果、缺失分组和有效生成样本量：`details/api/api_calibration.csv`。
- 8708次请求、8707次有效及失败角色：`details/api/api_failure_rates.csv` 和 `logs/api_replay.json`。
- 生成样本量100、200、500的单批描述：`details/api/api_generation_sizes.csv`。
- 条件回答类别频率：`details/api/api_category_frequencies.csv`；性别和年龄交叉群体：`api_group_means.csv`。
- 预测与真实类别的校准分箱：`details/api/api_probability_calibration.csv`。

表1、表3、表9以及图4由研究设计生成；表2来自文献，其余结果由本研究提供的真实调查数据、实际API回答和模拟程序计算。第三部分的男女各半、40%与20%幸福比例示例是明确标注的假设计算，不是新增实证数据；推导见 `docs/derivations.md`。
