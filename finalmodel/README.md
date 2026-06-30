# 最终提交模型说明

本目录保存课程作业最终提交模型 `GRU0` 及其评估脚本、配置文件和复核结果。根据 [report/main.pdf](/d:/讲义课件合集/多智能体基础/组队课题/MARL-Sokoban/MARL-project-sokoban/report/main.pdf) 的结论，`GRU0` 是当前实验中表现最稳定的最终提交模型。

## 模型信息

- 模型名称：`GRU0`
- 算法：HAPPO
- 架构：GRU encoder，不显式输入动作历史
- 训练环境：`7x7` 双人 `TwoPlayer-Sokoban`
- 主要评估脚本：`finalmodel/eval_finalmodel.py`

## 目录内容

本目录包含以下文件：

- `config.json`
- `models/actor_agent0.pt`
- `models/actor_agent1.pt`
- `models/critic_agent.pt`
- `models/value_normalizer.pt`
- `eval_finalmodel.py`
- 若干评估结果 `json`

## 来源实验

最终模型从以下长程训练结果中整理得到：

`results/happo-rnn-bfs-nodl-nopush-nohist-mixv012-r4/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_nohist_mixv012_r4/happo-rnn-bfs-nodl-nopush-nohist-mixv012-r4-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-28-23-55-40`

## 评估方法

在 `v0` 场景上进行评估：

```bash
python finalmodel/eval_finalmodel.py \
  --episodes 100 \
  --device cuda \
  --seed 50000
```

在配置文件指定的 `v0/v1/v2` 场景池上进行评估：

```bash
python finalmodel/eval_finalmodel.py \
  --episodes 90 \
  --device cuda \
  --seed 50000 \
  --scenario_pool config
```

若只需快速验证评估链路，可运行：

```bash
python finalmodel/eval_finalmodel.py \
  --episodes 5 \
  --device cpu
```

## 图形化 Demo 演示

在仓库根目录运行以下命令，可以一键启动最终模型的图形化推理演示：

```bash
./demo_finalmodel_gui.sh
```

该脚本会优先使用本机 `harl-sokoban` conda 环境，加载
`finalmodel/models/actor_agent0.pt` 和 `finalmodel/models/actor_agent1.pt`，
默认同时启动 9 个 `TwoPlayer-Sokoban` 场景，并以九宫格窗口展示推理过程。
演示中每隔 1 秒执行一次 agent 动作，场景结束后会自动重置继续展示。

常用参数：

```bash
./demo_finalmodel_gui.sh --device cpu
./demo_finalmodel_gui.sh --interval 0.5
./demo_finalmodel_gui.sh --headless --steps 10 --save-gif finalmodel/demo.gif
```

## 补充复核结果

### `v0` 单场景复核

以 `seed=50000` 对 `v0` 进行 20 个 episode 的确定性评估：

```bash
python finalmodel/eval_finalmodel.py \
  --episodes 20 \
  --device cpu \
  --seed 50000 \
  --output-json finalmodel/eval_v0_20ep_seed50000_metrics.json
```

结果如下：

- success rate：`0.4500`
- box completion ratio：`0.6500`
- final boxes on target：`1.3000`
- episode reward：`2.0850`
- episode base reward：`1.5350`

### `v0/v1/v2` 混合场景复核

以 `seed=50000`、`--scenario_pool config` 进行 30 个 episode 的确定性评估，每个场景 10 局：

```bash
python finalmodel/eval_finalmodel.py \
  --episodes 30 \
  --device cpu \
  --seed 50000 \
  --scenario_pool config \
  --output-json finalmodel/eval_mixed_30ep_seed50000_metrics.json
```

整体结果如下：

- success rate：`0.6000`
- box completion ratio：`0.7500`
- final boxes on target：`1.5000`
- episode reward：`7.3967`
- episode base reward：`6.7067`

分场景结果如下：

- `TwoPlayer-Sokoban-v0`：success `0.5000`，box completion `0.6500`
- `TwoPlayer-Sokoban-v1`：success `0.7000`，box completion `0.8500`
- `TwoPlayer-Sokoban-v2`：success `0.6000`，box completion `0.7500`
