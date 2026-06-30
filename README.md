# MARL Sokoban

本仓库为北京大学多智能体强化学习课程组队课题的提交代码与材料，任务为双人合作推箱子（Two-Player Sokoban）中的多智能体强化学习。项目基于 `gym-sokoban` 环境与 `HARL` 框架实现，围绕 HAPPO 主线，系统比较了 reward shaping、模型结构、输入历史、信用分配与混合环境训练等方向。

## 项目概述

本项目完成了以下核心工作：

- 将 `gym-sokoban` 中的双人环境封装为 HARL 可直接调用的多智能体环境。
- 在 HARL 中搭建并验证了 HAPPO 主线训练流程，同时保留 HASAC 对照入口。
- 实现并比较了多种 reward shaping、MLP/CNN/GRU 架构、动作历史输入、13x13 padding、credit assignment 与 HAHyPO 变体。
- 提供了训练脚本、评估脚本、可视化脚本、最终报告与最终提交模型。

## 主要结论

根据 [report/main.pdf](/d:/讲义课件合集/多智能体基础/组队课题/MARL-Sokoban/MARL-project-sokoban/report/main.pdf) 的实验结果，本项目的主要结论如下：

- 在固定 `7x7`、`2 boxes` 的双人 Sokoban 设置下，HAPPO 能够稳定学到协作策略。
- 去除 deadlock 与 pushability 负项的 `v1-nodl-nopush` 是当前实现中最稳的 MLP baseline。
- 语义上更精确的 reverse-push shaping v2 并未带来更优结果，说明“更正确”的奖励不一定更易优化；不可达距离的裁剪 cap 对训练稳定性十分关键。
- 在 `7x7` 设置下，GRU 表现最好；显式加入动作历史反而会降低 GRU 性能，说明“可以表示”不等于“容易优化”。
- 信用分配是重要问题，但当前 actor-side credit 的收益有限；HAHyPO 中引入 GRPO-style 相对优势也没有超过 HAPPO 主线。
- 在 `13x13` padding 设置下，`random padding` 几乎无法训练，`top-left padding` 虽仍困难但 CNN 明显最优；distillation 没有带来可靠提升。

## 仓库结构

- `gym-sokoban/`：原始 Sokoban 环境及其地图逻辑。
- `HARL/`：多智能体强化学习框架、环境接入、训练与评估代码。
- `run*/`：实验脚本与分组说明。
- `report/`：课程报告源码与编译后的 PDF。
- `finalmodel/`：最终提交模型、评估脚本与评估结果。

## 关键文件

- [report/main.tex](/d:/讲义课件合集/多智能体基础/组队课题/MARL-Sokoban/MARL-project-sokoban/report/main.tex)：课程报告源码。
- [report/main.pdf](/d:/讲义课件合集/多智能体基础/组队课题/MARL-Sokoban/MARL-project-sokoban/report/main.pdf)：课程报告 PDF。
- [HARL/harl/envs/sokoban/sokoban_env.py](/d:/讲义课件合集/多智能体基础/组队课题/MARL-Sokoban/MARL-project-sokoban/HARL/harl/envs/sokoban/sokoban_env.py)：HARL 侧 Sokoban 环境封装。
- [HARL/harl/envs/sokoban/reward_shaping.py](/d:/讲义课件合集/多智能体基础/组队课题/MARL-Sokoban/MARL-project-sokoban/HARL/harl/envs/sokoban/reward_shaping.py)：奖励塑形实现。
- [HARL/harl/algorithms/actors/happo.py](/d:/讲义课件合集/多智能体基础/组队课题/MARL-Sokoban/MARL-project-sokoban/HARL/harl/algorithms/actors/happo.py)：HAPPO actor 实现。
- [finalmodel/README.md](/d:/讲义课件合集/多智能体基础/组队课题/MARL-Sokoban/MARL-project-sokoban/finalmodel/README.md)：最终模型说明与评估方式。

## 环境准备与验证

Linux 服务器建议直接使用仓库内脚本完成环境部署：

```bash
cd /path/to/MARL-project-sokoban
bash HARL/scripts/setup_sokoban_linux.sh
conda activate harl-sokoban
python HARL/scripts/verify_sokoban_setup.py
```

如需指定 CUDA 对应的 PyTorch 轮子通道，可设置 `TORCH_CHANNEL=cu118`、`cu121`、`cu124`、`cu126` 或 `cpu`。

## 训练与评估

训练入口位于 `HARL/examples/train.py`，实验脚本集中在各 `run*/` 目录中。课程报告中主要使用的代表性脚本包括：

- `run8/`：主线 reward shaping 方案及续训实验。
- `run9/`：GRU / 动作历史相关实验。
- `run14/` 与 `run15/`：`13x13` padding 与混合环境实验。
- `run16/` 与 `run17/`：credit assignment 与 HAHyPO 对照实验。

最终提交模型位于 `finalmodel/`。评估示例：

```bash
python finalmodel/eval_finalmodel.py --episodes 100 --device cuda --seed 50000
```

若要在配置中的 `v0/v1/v2` 场景池上评估：

```bash
python finalmodel/eval_finalmodel.py --episodes 90 --device cuda --seed 50000 --scenario_pool config
```

## 结果与可视化

训练结果默认保存在运行目录下的 `results/` 中。可使用以下脚本对单次实验结果进行整理和绘图：

```bash
python HARL/scripts/plot_sokoban_metrics.py --run-dir <run_directory>
```

该脚本会读取 `summary.json` 或 TensorBoard event 文件，并生成奖励、优化指标与 Sokoban 专用指标的曲线图。

## 最终提交模型

最终提交模型为 `GRU0`，位于 `finalmodel/` 目录。该模型为不显式输入动作历史的 GRU 版本，在当前实验中表现最稳定，并被选为课程作业提交模型。模型来源、评估命令与补充结果请见 [finalmodel/README.md](/d:/讲义课件合集/多智能体基础/组队课题/MARL-Sokoban/MARL-project-sokoban/finalmodel/README.md)。
