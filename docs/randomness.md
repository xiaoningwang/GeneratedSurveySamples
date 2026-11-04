# 随机过程、软件与环境

## 原实验环境

原始环境文件保存在 `docs/evidence/original_environment_checked.json` 和 `original_simulation_environment.json`：Windows 11，系统标识 `Windows-11-10.0.26200-SP0`；CPython 3.13.7，64位；NumPy 2.3.5、pandas 2.2.3、SciPy 1.17.0、scikit-learn 1.8.0、Matplotlib 3.10.8。原检查还记录 python-docx 1.2.0，当前计算程序不依赖它。

`requirements_survey.txt` 固定上述五个计算包。每次重算自动记录实际 Python、操作系统、计算包版本、开始和结束时间，以及完整标准输出与错误输出。验证运行的环境保存在 `docs/evidence/verified_environment.json`，不替代原Windows环境。

## 随机数与顺序

| 用途 | 程序 | 原随机数设置 | 运行顺序及抽取方法 |
| --- | --- | --- | --- |
| 训练—评价划分 | `prepare_and_analyse.prepare` | `default_rng(20261012)` | 对659行做一次排列，前100行训练，其余559行评价 |
| 逻辑回归 | `prepare_and_analyse.fit_controls` | 默认确定性 lbfgs 求解 | `C=1.0, max_iter=2000`，年龄标准化、类别独热编码 |
| 随机森林与CART | 同上 | `random_state=20261012` | 森林200棵树、最大深度5、叶节点至少5条、`n_jobs=1`；CART深度5、叶节点至少5条。使用scikit-learn内部随机数机制 |
| 表5四总体均值模拟 | `simulate_means.py` | `default_rng(20261010)`，PCG64 | 按表4总体一至四顺序；同一随机流连续抽取，R=2000、n=200、m=5000 |
| 群体差值及样本量模拟 | `prepare_and_analyse.mechanisms` | `default_rng(20261014)`，PCG64 | 先按四总体顺序做群体实验，再做总体一样本量实验；不可将其拆成重新设种子的独立循环 |
| 传统非配对统计对照 | `finite_population_check` | `default_rng(20261013)`，PCG64 | 先题项，再n=50、100、200；从固定总体简单随机不放回抽取 |
| 统计模型三次二元生成 | `fit_models.run` | `default_rng(20261023)`，PCG64 | logistic、forest、cart依次；每个模型按agree、very_happy依次；按拟合概率抽3次伯努利回答 |
| 论文配对推断 | `paired_evaluation.paired_inference` | `default_rng(20261019+n)`，PCG64 | n=50、100、200，每种配置R=2000；相同n的不同预测方法使用同一校正样本序列 |
| API生成 | `collect_api.py` | 原请求未设置服务端seed | 固定提示、队列、模型参数并保存实际回答；分析读取原回答，不重新请求 |

四总体模拟先按目标组比例抽组别，再按组均值抽二元真实回答。生成回答按固定组配额 `m*q` 分别抽取二项计数。各方法共享本次真实样本与生成组均值。群体模拟中每次组内不足2条时不能构造方差，失败重复仍计入覆盖率分母。

有限总体推断固定训练数据和模型预测，从559条记录中简单随机不放回抽取校正样本；使用有限总体修正。少样本提示和统计模型另计100条训练标签，直接估计对照按n或n+100匹配预算。此次评价条件于已经固定的训练与预测，不在每次重复中重训或调用API。

## 重算结果的判断

`python run_paper.py --output-dir reproduction_results --verify` 保留原始输入，重算14张表、5幅图及正文数值，并生成 `logs/verification.json`。表格按论文显示精度逐单元格比较；正文完整精度数值使用 `rtol=atol=1e-12`。图的数值来源与绘图程序逐一列在 `docs/result_map.md`；不同字体或渲染版本导致的图片字节差异不作为统计数值差异。

若更换系统后统计模型出现数值差异，核对报告会列出具体单元格并返回失败，不会用固定表格覆盖新计算值。重新采集API数据属于新实验，不能用于声称原论文回答被逐条重现。
