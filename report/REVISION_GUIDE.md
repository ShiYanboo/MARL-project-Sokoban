# 报告修改说明文档

本说明文档用于指导后续对 [main.tex](/d:/讲义课件合集/多智能体基础/组队课题/MARL-Sokoban/MARL-project-sokoban/report/main.tex) 的定向修改。目标不是重写整篇报告，而是在保持前半部分写法、排版风格和叙述基调的前提下，统一后半部分的结论与表达。

项目结构可以用四句话概括：

- 仓库主体分为 `gym-sokoban/`、`HARL/`、`run*/` 与 `report/` 四层。
- `gym-sokoban/` 提供原始双人 Sokoban 环境，`HARL/` 提供 HAPPO/HASAC 与训练框架。
- `run*/` 和 `plots/` 记录实验入口与出图结果。
- `report/main.tex` 是正式报告源文件，`report/main.pdf` 是当前排版参考。

## 1. 编辑边界

本轮只允许修改以下内容：

- `report/main.tex` 中 **3.3 之后**的正文。
- 所有 `figure` 的 caption。

本轮不要修改以下内容：

- 第 1 节到第 3.2 节
- 图片文件本身
- 大范围排版和 float 位置
- 表格 caption 与代码 listing caption

例外有两类：

- `abstract` 需要在全文后半部分定稿后按最终结论整体更新，尤其要与主结果、模型架构、mix/padding 和最终模型选择保持一致。
- 如果某个表格 caption 或 listing caption 与新的结论表述直接冲突，可以做最小必要修改；否则一律不动。

请始终记住，这次任务不是“重写整篇报告”，而是“在保持前半部分写法和排版风格的前提下，统一后半部分结论与表达”。

## 2. 全局写作要求

文风要求如下：

- 以中文为主，可中英文混写。
- 句子保持简练、书面、结论导向。
- 语气尽量像课程报告，而不是实验记录或调试日志。
- 多用“结果表明 / 实验显示 / 说明 / 可能原因是”等表达。
- 少用“我们觉得 / 可能是 bug / 试了很多次才发现”等口语化或过程化说法。

内容筛选要求如下：

- 只保留对正式报告有展示价值的信息。
- 不写调试史，不写实验期内部命名逻辑，不写无关试错过程。
- 不把 `runX` 系列当成正文叙述主线。

以下内容在正式报告中必须全部去掉：

- “伪 mix”
- “pseudo-mix”
- “伪混合”
- “这是实现错误导致的”
- 任何等价于“这组实验其实和 v0 一样，只是后来才发现”的说法

你只需要知道一件事：这些信息对实验内部排查有用，但不应该出现在形式完整的正式报告里。

## 3. 需要改写的现有结构

当前 `main.tex` 中，后半部分主要从以下位置开始：

- `\subsection{信用分配与 Actor-Side Credit}`
- `\subsection{HAHyPO and GRPO-Style Advantage}`
- `\section{实验组织}`
- `\section{结果与分析}`
- `\section{结论与启发}`
- `\section{复现说明}`

本轮改写的目标不是完全推翻这些结构，而是在此基础上收束口径、简化叙述，并保证后半段与前半段风格一致。

## 4. 逐节改写要求

### 4.1 3.3 信用分配

这一节改成短节即可，重点只保留两点：

- turn-based 设置下，credit assignment 是真实存在的问题。
- 当前 actor-side credit 没有给出决定性提升。

写法要求：

- 语气保守，不夸大。
- 不展开太多实现细节。
- 不写成长篇算法说明。
- 不要把这一节写成“提出了一个已经验证有效的方法”，因为目前结论并不支持。

### 4.2 3.4 HAHyPO / GRPO-style advantage

这一节保留单独章节，但核心结论要改明确：

- GRPO 风格项的引入没有带来性能提升。
- 这与语言模型中的使用场景不同。
- 其可能原因是，GRPO 的组内相对排序信号较粗，难以替代 Sokoban 这种稀疏、长时序、强 credit assignment 依赖任务中的逐步优势估计。
- HAHyPO 可以作为探索性变体保留，但不能替代 HAPPO 主线。

写法要求：

- 不要把重点放在“实现了一个新算法入口”上。
- 重点应放在“为什么没有比主线更好”。

### 4.3 实验组织

这一节要精简，不要继续把所有 `runX` 的历史都写出来。

保留原则：

- 对读者真正有用的主线标签可以点名。
- `v1-nodl-nopush` 这类关键 baseline 必须明确点名。
- 其他内容尽量按系列级别概括。

推荐保留的系列表达方式：

- architecture/history series
- mix/padding series
- HAHyPO series
- credit series

不建议写进正文的内容：

- 临时试验
- 调参碎片
- 与主结论无关的 `runX`
- 只对内部复现有价值、对读者没有解释收益的脚本细节

### 4.4 结果与分析

这一节需要重组为几个固定的大块，顺序不要改。

#### 4.4.1 主结果

只比较：

- HAPPO baseline
- `v1-nodl-nopush`

主结论必须明确写成：

- `v1-nodl-nopush` 是当前实现中最稳的 baseline。

本节不要写：

- 训练史
- pseudo-mix 相关叙述
- 无关变体的横向堆砌

#### 4.4.2 距离语义正确性的影响

本节只围绕：

- v1
- v2
- `v2-clip` / cap

必须得出的结论：

- 语义更精确不一定更好。
- 更精确的 reward 可能带来更崎岖的 optimization landscape，因此更难学。
- `v2-clip` / cap 很重要，因为它关系到信号尺度是否可控。

#### 4.4.3 模型架构的影响

本节只围绕原始 7x7 实验。

结论固定为：

- GRU 效果最好。
- 动作历史会让 GRU 性能下降。
- `mlp-8` 比 `gru-8` 更差。

解释口径固定为：

- 非 Markov 环境可以通过历史输入被 Markov 化，因此从表示能力上并非完全做不到。
- 但“可以表示”不等于“容易优化”。
- 额外历史会增加学习难度，因此效果提升不来自动作历史本身，而更可能来自 GRU 的时序建模能力。

#### 4.4.4 Credit assignment

这一块单开一个小节即可，少说两句。

只表达：

- 问题重要。
- 当前改动收益有限。

不要展开成长段。

#### 4.4.5 HAHyPO

这一块单开一节。

结论固定为：

- GRPO 的引入没有提升性能。

需要附带一句分析：

- 其弱点在于轨迹级相对排序信号过粗，难以替代逐步 advantage。

注意：

- 不要把它简单写成“和语言模型一样有效”。
- 也不要写成“完全无意义”；更准确的表述是“作为探索性方向可保留，但不构成主线改进”。

#### 4.4.6 Mix

这一块拆成两个子块或两个自然段：

第一部分写 `random padding`：

- 几乎无法训练。
- 原因写成“有效信号稀疏，且模型需要在 7x7 基础上额外学会平移不变性”。

第二部分写 `top-left padding`：

- 仍然困难，但 CNN 最优。
- 这一结果显著体现了 CNN 的空间归纳偏置与平移不变性优势。
- 可以加入一个可延展的观察：在原始等大输入实验中，曲线通常很快离开起始阶段，因此早期趋势对最终表现有较强指示；但在 13x13 top-left padding 中，训练会出现明显更长的平台期，random padding 的 warmup 平台可能更长。这个现象值得强调：padding 并没有增加有效任务信息，只是扩大了观测维度和网络输入容量，但模型仍然需要通过训练学会哪些位置是无信息的。对观察者而言，这相当于只能通过贝叶斯式的经验推断判断“某些输入区域不会带来信息”。因此，仅扩充输入规模或模型容量，即使不增加有效信息，也可能显著提高优化难度；容量提升通常需要与更丰富、更直接的训练信号匹配。正式报告中只分析 20M 以内的平台期即可，不展开 mix 阶段之后的曲线跳变。

`distillation` 可以提，但结论固定为：

- distillation 没有效果。
- 很可能受限于模型容量。

#### 4.4.7 无用动作惩罚项

这一小节可以作为辅助分析写入，但不要写成主线改进。

核心结论固定为：

- 无用动作惩罚项的作用主要是加速早期行为探索，而不是提供与任务完成直接相关的正向信号。
- 结果显示，该惩罚确实降低了 useless-action rate，并提高了 agent 的移动次数；这说明奖励项改变了底层行为分布，使 agent 从“不动 / 浪费动作”转向“更多行动”。
- 但成功率与箱子到位率没有超过 `v1-nodl-nopush`，长程潜力反而较弱。这说明限制性能的主要因素并不是“没有合理动作”或“动作不够活跃”，而是推箱方向选择、长期规划和避免错误 push。
- 因此，该实验应解释为：无用动作惩罚验证了低层行为先验可以减少无效动作，但多动不等于会解 Sokoban。

推荐写法：

> Useless-action penalty effectively reduces invalid or semantically useless behaviors and increases agent movement, especially with larger penalty weights. However, this improvement in activity does not translate into higher success rate. The result suggests that early exploration is not primarily limited by the absence of reasonable primitive actions; rather, the bottleneck lies in choosing useful push directions and avoiding irreversible bad box placements.

不可接受的处理方式：

- 把它写成比 `v1-nodl-nopush` 更好的方法。
- 只根据初期短暂领先得出“最终性能提升”的结论。
- 把 useless-action penalty 解释成提供了明确的 task-progress 正向奖励。

### 4.5 结论与启发

这一节要和新的“结果与分析”完全对齐。

要求如下：

- 只总结正文已经支撑的结论。
- 删除 pseudo-mix、调试史、无充分结果的小分支相关语句。
- 顺序尽量与“主结果 → 距离语义 → 模型架构 → credit → HAHyPO → mix”一致。

### 4.6 最优模型

在结尾新增一个编号章节，放在：

- `复现说明` 之后
- `致谢` 之前

本节写成最终提交模型说明。当前最终提交版本确定为 `GRU0`，即不显式输入动作历史的 GRU 版本。建议包含以下字段：

- 模型名称：`GRU0`
- 环境设置
- reward shaping 设置
- 架构与历史长度：GRU encoder，动作历史长度为 0
- 训练步数
- 选用理由
- 待提交 checkpoint / 指标占位

这一节可以先不填具体 checkpoint 路径和最终数值，但必须明确最终提交模型版本是 `GRU0`。选用理由应与正文结论一致：GRU0 在 7x7 主线中整体更稳，且没有动作历史输入带来的额外优化负担。

最终 GRU0 的 eval 成功率、箱子到位率及其 raw/smoothed 口径说明已经记录在同目录下的 `finalgru.md`。填写“最优模型”章节时，以该文件中的 final raw evaluation 指标为准，并说明报告图像经过 smoothing，视觉上可能低于最后一次 raw eval 点。

最终模型 checkpoint 已单独整理在仓库根目录的 `finalmodel/` 下。该目录包含 `models/`、`config.json` 和独立评测脚本 `eval_finalmodel.py`。如需在 v0 环境上复测最终模型，可运行：

```bash
/home/ybshi/miniconda3/envs/harl-sokoban/bin/python finalmodel/eval_finalmodel.py \
  --episodes 100 \
  --device cuda \
  --seed 50000
```

如果需要复现训练配置中的 v0/v1/v2 多环境平均评测，可使用 config 中保存的 scenario pool：

```bash
/home/ybshi/miniconda3/envs/harl-sokoban/bin/python finalmodel/eval_finalmodel.py \
  --episodes 90 \
  --device cuda \
  --seed 50000 \
  --scenario_pool config
```

脚本会同时输出 overall 指标和 per-scenario 指标。当前已有一次 30-episode mixed re-evaluation 结果保存在 `finalmodel/eval_mixed_30ep_seed50000_metrics.json`，overall success rate 为 `0.6000`，overall box completion ratio 为 `0.7500`。

快速 smoke test 可运行：

```bash
/home/ybshi/miniconda3/envs/harl-sokoban/bin/python finalmodel/eval_finalmodel.py \
  --episodes 5 \
  --device cpu
```

说明：不加 `--scenario_pool` 时，该脚本执行的是 `TwoPlayer-Sokoban-v0` 上的独立 deterministic evaluation；加 `--scenario_pool config` 时，它执行的是 v0/v1/v2 混合 deterministic evaluation。若报告中引用此脚本结果，需要写清楚 v0-only 或 mixed-scenario 的具体口径。

## 5. 图注修改规则

所有 `figure` caption 都需要按同一风格重写。

统一句式建议：

- 第一分句交代设置或比较对象。
- 第二分句交代主要现象。
- 第三分句给出唯一核心结论。

图注总要求：

- 简洁。
- 不写内部训练流程词汇。
- 不写 pseudo-mix。
- 不写调试过程。
- 不塞过多 `runX` 细节。

避免使用的表达：

- “最好、最强、显著领先所有方法”这类绝对措辞

除非图中趋势非常明确，否则建议改成：

- “更稳”
- “更容易优化”
- “表现更好”
- “显示出优势”

各图的结论中心应当如下：

- 主结果图：只保留 HAPPO baseline 与 `v1-nodl-nopush` 的结论中心。
- v2 图：突出语义更强但更难学，cap 很重要。
- architecture/history 图：突出 GRU 优势与动作历史副作用。
- HAHyPO 图：突出 GRPO-style 项未带来提升。
- credit 图：突出收益有限。
- random/top-left 图：突出 random 更难、CNN 最优。
- distill 图：突出 distillation 无法可靠迁移。

注意：

- 本轮优先改 `figure` caption。
- `table` caption 和 `listing` caption 原则上不动，除非与上述结论直接冲突。

## 6. Run 标注策略

正式报告中的 `runX` 信息必须压缩到最少。

正文只保留少数有明确解释价值的标签，例如：

- `v1-nodl-nopush`
- `GRU-0`
- `GRU-8`
- `CNN`
- `v2-clip`
- `top-left padding`
- `random padding`
- `distillation`

不要做的事：

- 不把所有 `runX` 都翻译进正文。
- 不解释每个 `runX` 对应哪个临时试验。
- 不让 `runX` 成为主叙事线索。

更合适的做法是：

- `runX` 留在内部记录、脚本命名和复现材料里。
- 报告正文只保留读者一眼能理解的语义标签。

## 7. 建议的后半段结构

如果需要对 3.3 之后的小节做轻度整理，建议结构按下面执行：

1. `3.3 信用分配`
2. `3.4 HAHyPO / GRPO-style advantage`
3. `4 实验组织`
4. `5 结果与分析`
5.1 主结果  
5.2 距离语义正确性的影响  
5.3 模型架构的影响  
5.4 Credit assignment  
5.5 HAHyPO  
5.6 Mix  
5.7 无用动作惩罚项（作为行为先验辅助分析略写）
5. `6 结论与启发`
6. `7 复现说明`
7. `8 最优模型`

说明：

- 这里的编号是逻辑结构建议，不要求你现在强行重排前半段。
- 如果现有章节名基本可用，可以只改文字内容，不一定必须大改章节编号。

## 8. 验收清单

修改完成后，请逐项自查：

- `report/main.tex` 的实质改动只发生在 3.3 之后与 `figure` captions。
- 所有 `figure` captions 都已改写，但图位置基本未动。
- 后半部分不再出现“伪 mix / pseudo-mix / 伪混合”。
- 结果分析严格按 6 个大块组织：主结果、距离语义、模型架构、credit、HAHyPO、mix。
- 无用动作惩罚项只被写成行为先验 / 早期探索分析，而不是主线性能提升。
- `runX` 信息被压缩到必要最少。
- `abstract` 已根据全文结论整体更新，没有继续沿用旧结论。
- 新增了“最优模型”章节，并明确最终提交模型版本为 `GRU0`。
- 全文语气与前半部分一致，简练、书面、结论导向。

## 9. 默认假设

除非有新的明确指令，否则默认以下前提成立：

- 前半部分（不含 abstract）已经定稿，不回头重写；`abstract` 需要根据最终全文结论同步更新。
- 即使当前 `main.tex` 里仍残留旧说法，本说明文档也不要求回改 1--3.2。
- “所有图的脚注”按 `figure` caption 理解，不包含 `table` caption 和代码 `listing` caption。
- “最优模型”可以暂不填写最终 checkpoint 或数值，但必须写明提交版本为 `GRU0`。
