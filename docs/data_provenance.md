# 数据来源与处理

## 原始输入与获取记录

本研究使用 Xie 等（2026）团队 SSDataBench-Data 仓库公开的 **GSS 2018 处理后1,000条子样本**，没有另行组织真人调查。该文件是本文数据加工的起点，不是 NORC 发布的完整 GSS 原始数据库。

- 文件精确地址：<https://raw.githubusercontent.com/LemengLiang/SSDataBench-Data/main/processed_survey_data/sampled_gss.csv>
- 文件查看页：<https://github.com/LemengLiang/SSDataBench-Data/blob/main/processed_survey_data/sampled_gss.csv>
- 本地原文件：`survey_data/sampled_gss.csv`。
- SHA-256：`7b86f23d24b3355eb2094ccee1a5693e8950d55fef6db875d3c4bdb82e1232dc`。
- 获取方式：直接读取上述已发布 CSV；不涉及数据库检索条件或重新从完整 GSS 抽取1,000条记录。
- 时间证据：2026年10月13日 API 实验已使用该数据及固定划分。原始清单没有记录首次下载日期；作者投稿前须依据实际下载记录补充确切获取日期，不能把 GSS 调查年份、API 日期或此次复算日期替代下载日期。
- 上游处理说明保存在 `survey_data/source_notes/`。`GSS_preprocessing_author.py` 和 `gss_code_config_author.yaml` 用于解释上游变量编码，不在本研究离线流程中运行，也不代表本研究重新执行了上游原始数据库加工。

原始数据与处理说明应按其上游来源引用；本仓库不另行改变第三方数据的使用条件。

## 从1,000条记录到最终分析数据

`survey_code/prepare_and_analyse.py::prepare()` 执行以下步骤，并将中间数据保存为 `details/complete_cases.csv`：

1. 对 `gender、age、education、marital_status、laborforce、gender_role_attitude、happy` 七列做完整案例筛选，得到659条记录。其余未使用字段的缺失不触发删除。
2. 定义 `agree = 1(gender_role_attitude == 'Agree')`。上游已经将 GSS `fefam` 原代码1、2合并为 Agree，3、4合并为 Disagree；本研究不把文件中 Agree 误当作原四级回答中的单个类别。
3. 定义 `very_happy = 1(happy == 'Very happy')`，其余两个有效幸福类别为0。
4. 保留原 `profile_id`，转成整数。按性别与年龄是否达到60岁构造 `group`。
5. 用 `numpy.random.default_rng(20261012).permutation(659)` 打乱完整案例的**现有行顺序**。前100条为训练集，其余559条为评价集。
6. 与 `survey_data/training.csv`、`survey_data/benchmark.csv` 逐行核对；程序不会另抽一份训练或评价数据覆盖原始划分。

年龄89是“89岁及以上”的顶编码。教育、婚姻和劳动参与沿用上游分类；例如上游将 Married 与 Separated 归入 Married，此处不重新定义分类。变量完整说明及实际取值见 `docs/variable_dictionary.csv`。

本文比例、群体差值和区间评价针对固定的559条评价记录构成的经验有限总体。100条训练记录不进入该总体的分母。计算不使用全国调查权重。

## 大语言模型实验记录

- 2026年10月13日，API 请求与返回模型标识均为 `deepseek-flash`；原始响应中的模型标识和时间戳可直接核对。
- 思考模式 disabled，温度0.7，输出上限1,024 token；每次新会话同时回答两道英文题项。
- 四种提示：plain（普通指令）、expert（调查专家提示）、respondent（受访者角色提示）、fewshot（少样本提示）。少样本提示包含同一100条训练记录。
- 条件生成：559个固定画像 × 4种提示 × 3次回答 = 6,708次请求。
- 联合生成：每种提示500次，共2,000次请求。正式合计8,708次，8,707次有效。
- 试运行32次全部有效，与正式分析分开保存，不计入正式结果。
- 正式唯一失败请求为 `b5f38fe3000ec3443bf379ffcf8feeeaa2a755de9f0e80df9793cc3ec2a31e96`。模型将五项画像放入嵌套对象，违反顶层字段要求。按原规则记为 `invalid_output`，未修复、未补抽。普通指令联合生成按499条有效记录计算描述指标；500条批次保留不完整状态。

原始响应原字节保存在 `survey_data/api_raw/original_responses.zip`；main、pilot分目录。每条记录保留提示词、响应正文、解析结果、耗时、重试及token用量。计划与结束状态同目录保存。`api_metadata.json` 是分析用计划，`api_records.csv` 是正式响应的扁平分析表。

离线程序逐条重建提示和请求身份哈希，重新解析响应，核对8,708行分析记录。原实验在Windows上将少样本CSV写为CRLF；采集代码现显式固定该换行符，保证跨系统重建相同提示。请求内容、参数、身份哈希及原回答均未改变。

原始计划中的 `prompt_code_sha256` 是当时采集脚本的指纹；整理后的脚本路径及导入发生过变化，因此以逐条重建实际提示、请求身份及原始响应校验作为内容核对，不冒充脚本字节完全相同。

## 文献输入与本研究输出

表2使用 `survey_data/published_inputs.json` 中 Xie 等（2026）原表1数值，代码只计算提高的百分点。表1、表3、表9是方法说明表。其余表及图的来源见 `docs/result_map.md`。文献案例不作为本研究模拟或API实验的观测结果。
