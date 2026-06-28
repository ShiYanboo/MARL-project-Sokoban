# MARL Sokoban 项目说明

## 1. 当前进度

目前已经完成以下工作：

- 已将 `gym-sokoban` 的双人环境接入 `HARL`
- 已搭好 `HAPPO` 和 `HASAC` 两套训练框架
- 已补好本机验证脚本与 Linux `conda` 安装脚本
- 已完成最小可运行验证，确认训练、评估、保存模型的链路打通
- 已补充结果可视化脚本，可从 `summary.json` 或 TensorBoard event 文件直接画曲线图
- 已加入 reward shaping v2，并保留旧版朴素 BFS shaping 的命令行开关
- 已加入 v0/v1/v2 混合训练支持，可按 `scenario_pool` 轮换不同 Sokoban 变体
- 已加入 GRU/RNN 实验入口，支持把过去若干步双方实际执行动作加入 observation
- 已加入 13x13 padding / true-mix 实验入口，以及 actor-side credit assignment 小型算法改动

当前重点不是“已经训出很强的策略”，而是“实验框架已经可复现、可扩展、可继续做大规模实验”。目前主线实验仍以 HAPPO + MLP + 旧 BFS reward shaping 为主，CNN、GRU/RNN、13x13 padding 和 actor-side credit 都保留为对照。

截至当前阶段，实验结论大致是：

- 单纯换 CNN encoder 没有带来稳定提升。
- GRU/动作历史在 7x7 上看起来更有希望，但 13x13 padding 和 true-mix 泛化仍然困难。
- reward shaping v2 在语义上更接近 Sokoban 推箱约束，但实际效果不如旧的朴素 BFS shaping 稳。
- 目前最稳的轻量方案仍是 `v1-nodl-nopush`：保留 box-target BFS 势函数和 agent-box 势函数，去掉 deadlock 与 pushability 负项。
- 新增 `run16/` 只做一个很小的算法层 credit-assignment 尝试。如果这组 5M 短实验仍没有明显超过 `v1-nodl-nopush`，建议停止继续魔改算法，转入报告整理。

---

## 2. 项目结构

项目主要分成两部分：

- `gym-sokoban/`
  原始 Sokoban 环境仓库，负责地图生成、推箱子规则、原始奖励、渲染
- `HARL/`
  多智能体强化学习训练框架，负责算法、buffer、runner、日志、训练与评估

这次工作的核心，是把 `gym-sokoban` 的 `TwoPlayer-Sokoban-v0` 封装成一个符合 HARL 接口的 `sokoban` 环境，从而直接复用 HARL 里的 HAPPO 和 HASAC。

---

## 3. 关键代码位置

### 3.1 Sokoban 环境对接

- `HARL/harl/envs/sokoban/sokoban_env.py`
  HARL 侧的 Sokoban 环境封装，负责 observation/action/reward/info 的转换
- `HARL/harl/envs/sokoban/reward_shaping.py`
  Sokoban reward shaping 工具，负责 BFS 距离、pushability、deadlock 和 agent-box 距离等辅助奖励
- `HARL/harl/envs/sokoban/sokoban_logger.py`
  Sokoban 的 on-policy 日志器，负责额外记录成功率、回合制执行统计、推箱次数等指标
- `HARL/harl/configs/envs_cfgs/sokoban.yaml`
  Sokoban 环境配置
- `run8/`
  朴素 BFS shaping 已筛选方案的续训、v0/v1/v2 混合训练脚本
- `run9/`
  GRU/RNN + 动作历史 observation 的新实验脚本

HARL 注册位置：

- `HARL/harl/utils/envs_tools.py`
- `HARL/harl/utils/configs_tools.py`
- `HARL/harl/envs/__init__.py`
- `HARL/examples/train.py`

原始环境逻辑位置：

- `gym-sokoban/gym_sokoban/envs/sokoban_env.py`
- `gym-sokoban/gym_sokoban/envs/sokoban_env_two_player.py`
- `gym-sokoban/gym_sokoban/envs/sokoban_env_variations.py`

### 3.2 HAPPO

- `HARL/harl/algorithms/actors/happo.py`
- `HARL/harl/runners/on_policy_ha_runner.py`
- `HARL/harl/algorithms/critics/v_critic.py`
- `HARL/harl/configs/algos_cfgs/happo.yaml`

### 3.3 HASAC

- `HARL/harl/algorithms/actors/hasac.py`
- `HARL/harl/runners/off_policy_ha_runner.py`
- `HARL/harl/algorithms/critics/soft_twin_continuous_q_critic.py`
- `HARL/harl/algorithms/critics/twin_continuous_q_critic.py`
- `HARL/harl/models/value_function_models/continuous_q_net.py`
- `HARL/harl/configs/algos_cfgs/hasac.yaml`

---

## 4. 环境封装逻辑

原始 `TwoPlayer-Sokoban` 本质上还是“一个底层环境，每步只真正执行一个玩家动作”。HARL 需要的是标准多智能体接口，所以在 `HARL/harl/envs/sokoban/sokoban_env.py` 里做了以下封装：

- 两个 agent 各自输出一个 `Discrete(9)` 动作
- 9 个动作分别是 `noop + 4 push + 4 move`
- 当前默认控制方式是 `turn_based`
- 每一步只有一个 agent 处于激活状态，另一个 agent 只有 `noop` 可选
- 激活 agent 按回合在两个玩家之间交替
- 底层环境返回的是团队共享 reward，再复制给两个 agent
- 可选 `scenario_pool` 会在 reset 时按顺序轮换多个 gym scenario，例如 v0/v1/v2 做 1:1:1 混合训练
- 可选 `action_history_len` 会把过去若干步双方实际执行动作加入 observation；默认 `0`，旧脚本维度不变

当前 `info` 里额外补了这些统计，便于监测：

- `boxes_on_target`
- `box_completion_ratio`
- `steps_used`
- `step_reward`
- `action_moved_box`
- `action_moved_player`
- `had_conflict`
- `invalid_action_attempt`
- `noop_executed`
- `success`

---

## 5. 奖励设计与 Credit Assignment

### 5.1 当前奖励设计

底层原始奖励保持不变，定义位置：

- `gym-sokoban/gym_sokoban/envs/sokoban_env.py`

当前规则是：

- 每步惩罚：`-0.1`
- 箱子被推到目标点：`+1`
- 箱子离开目标点：`-1`
- 所有箱子都到目标点：默认 `+10`，可通过 `--reward_finished` 覆盖

HARL 封装层在原始奖励之上增加了可配置的 reward shaping，定义位置：

- `HARL/harl/envs/sokoban/reward_shaping.py`

早期默认权重较小，当前更推荐把正向势函数信号略微放大。`run6_0.sh` 和后续 `run6.*.sh` 系列使用的当前主线权重为：

- 箱子到目标的墙约束 BFS 最小匹配距离差分：权重 `0.20`
- 箱子可推动方向总数差分：权重 `0.05`
- 非目标位置且 `pushability=0` 的死锁箱惩罚：`0.8`
- 箱子到最近 agent 的可达距离和差分：权重 `0.10`

默认 `use_reward_shaping=False`，所以旧脚本仍严格使用原始奖励。传入 `--use_reward_shaping True` 后，最终奖励为 `base_reward + shaping_reward`，并仍作为团队共享奖励发给两个 agent。四项分别可通过 `--distance_shaping_weight`、`--pushability_shaping_weight`、`--deadlock_penalty` 和 `--agent_box_distance_shaping_weight` 覆盖；全部设为 `0` 也可恢复原始奖励。

`run5.sh` 及当前 `run6*` 系列使用 `--reward_finished 20`，只提高完成整局时的终局奖励；其他旧脚本未指定该参数时仍为 `+10`。

`--deadlock_penalty_mode state` 会对每个死锁状态每步惩罚；`increase` 只在死锁箱数量增加时惩罚一次，通常更适合训练早期探索。当前主线脚本使用 `increase`，避免同一死锁状态被每步重复重罚到 `-100` 量级。

重复推入 target 的次数感知奖励暂未加入。它依赖 episode 历史计数，而当前无 RNN actor 无法直接观察该状态，会额外引入部分可观测性。

### 5.2 reward shaping v2 与旧 BFS 兼容模式

当前 `reward_shaping.py` 同时支持两套距离语义：

- `--box_target_distance_mode reverse_push`
  v2 默认模式。箱子到目标距离使用 reverse-push distance table，估计“箱子理论上最少被推几次能到目标”。它考虑墙和推箱站位约束，但不把其他箱子作为长程障碍。
- `--box_target_distance_mode bfs`
  旧版朴素模式。箱子到目标距离使用目标出发的普通 BFS，只考虑墙，不考虑玩家和其他箱子。
- `--agent_box_distance_mode useful`
  v2 默认模式。agent-box 距离改为玩家到“能让全局 box-target 距离下降的有效站位”的距离。
- `--agent_box_distance_mode adjacent`
  旧版朴素模式。玩家到箱子相邻格的 BFS 距离，BFS 时会把箱子和另一名玩家作为障碍，并在相邻格距离上加 1。该模式用于复现实验早期的朴素 BFS shaping。

默认配置是 v2：

```yaml
box_target_distance_mode: reverse_push
agent_box_distance_mode: useful
```

`run8/` 中所有旧 BFS 续训脚本都会显式传入：

```bash
--box_target_distance_mode bfs
--agent_box_distance_mode adjacent
```

这样它们继续训练旧模型时不会悄悄切换到 v2 reward 语义。

另有 `--shaping_unreachable_distance` 可选参数，用于把不可达距离裁剪到指定值；默认 `~` 表示仍使用地图面积作为不可达距离。该参数主要用于 v2 诊断，例如 `run7.8/run7.9` 系列曾尝试 cap 到 5 或 12。

### 5.3 当前 credit 如何分配

默认环境层面不做手工细粒度 credit shaping，也就是说：

- 两个 agent 收到相同 reward
- 不直接指定“这一步到底该主要奖励哪个 agent”

credit assignment 主要由算法本身完成。

#### HAPPO 中的 credit

- 共享团队回报
- 中心化 critic 学习全局价值
- actor 按顺序更新，并通过 `factor` 修正前面 agent 更新带来的影响

对应位置：

- `HARL/harl/runners/on_policy_ha_runner.py`
- `HARL/harl/algorithms/actors/happo.py`

#### HASAC 中的 credit

- 共享团队回报
- twin Q critic 读取共享状态和 joint action
- actor 通过 critic 的 Q 值反馈，间接学习自己动作的贡献

对应位置：

- `HARL/harl/runners/off_policy_ha_runner.py`
- `HARL/harl/algorithms/critics/soft_twin_continuous_q_critic.py`

### 5.4 run16：actor-side credit assignment

当前新增了一个默认关闭的小型算法改动，用来测试“只给 active agent 更直接的 actor 学习信号”是否能降低方差。

动机如下：

- Sokoban 已经改成严格 `turn_based`。
- 每一步只有 active agent 的动作真正改变底层环境。
- inactive agent 只有 `noop` 可选，并且 actor buffer 中 active mask 为 `0`。
- critic 仍应学习团队回报，因为任务目标仍是协作完成关卡。
- 但是 actor 的 advantage 来自稀疏、长程、团队共享 return，方差很高；active agent 很难知道当前这一步移动/推箱到底是不是好。

因此 run16 没有改 critic 目标，也没有把环境 reward 改成个人奖励，而是新增了一个可选的 actor-side auxiliary credit：

```bash
--actor_credit_mode none|active_progress|active_reward
--actor_credit_coef 0.0
```

默认：

```bash
--actor_credit_mode none
--actor_credit_coef 0.0
```

这时所有旧实验行为完全不变。

开启后，runner 会从 `info` 中为每个 agent 生成一个 `credit_rewards` buffer，并在 HAPPO actor 更新前把它加到该 agent 的 actor advantage 上：

```text
actor_advantage = team_advantage + actor_credit_coef * active_agent_credit
```

critic buffer 仍然使用 `team_reward`，也就是原始团队奖励：

```text
critic_target_reward = team_reward
```

当前支持两种 credit：

- `active_progress`
  只在 agent 是本步 active agent 时生效。它给正向进展一个小 bonus，例如正 base reward、useful-push distance delta、正的 agent-box shaping、以及成功移动箱子的小奖励。
- `active_reward`
  只在 agent 是本步 active agent 时，把 team reward 的一个小比例加到 actor advantage 上。

对应代码：

- `HARL/harl/common/buffers/on_policy_actor_buffer.py`
  新增 `credit_rewards`
- `HARL/harl/runners/on_policy_base_runner.py`
  从 `infos` 生成 actor credit，并保证 EP critic 仍使用 `team_reward`
- `HARL/harl/runners/on_policy_ha_runner.py`
  在 actor update 前把 credit 加到 advantage 上
- `HARL/harl/envs/sokoban/sokoban_env.py`
  在 `info` 中记录 `team_reward`

这不是严格无偏的原始 policy gradient，而是一个很小的辅助学习信号。它的意义是测试“降低 active actor 的信用分配方差”是否比继续改 reward shaping / 网络结构更有帮助。推荐只跑 `run16/` 的三个 5M 短实验；如果没有明显收益，不建议继续扩展。

---

## 6. 训练结果、保存位置与可视化

### 6.1 结果保存在哪里

训练结果默认保存在“当前启动目录下的 `results/`”。

也就是说：

- 如果你在仓库根目录执行 `python HARL/examples/train.py ...`，结果会在 `./results/`
- 如果你先 `cd HARL` 再执行 `python examples/train.py ...`，结果会在 `HARL/results/`

路径结构大致是：

- `results/<env>/<scenario>/<algo>/<exp_name>/seed-xxxxx-YYYY-MM-DD-HH-MM-SS/`

如果命令行传入 `--run_name_prefix happo-base`，最后一级目录会自动携带关键参数，例如：

- `results/happo-base/<env>/<scenario>/<algo>/<exp_name>/happo-base-steps10000000-len150-env0-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-YYYY-MM-DD-HH-MM-SS/`

也就是说 prefix 同时作为 `results/` 下的分组目录，TensorBoard 可以只加载 `results/happo-base`。其中 `env0` 来自 `CUDA_VISIBLE_DEVICES=0`。前缀中包含 `hasac` 时，还会追加 HASAC 的温度/熵系数字段，例如 `entrocoe0.005`；开启自动温度时写作 `entrocoeauto`。不传 `--run_name_prefix` 时仍使用原来的路径和 `seed-时间戳` 命名。

注意：结果不会覆盖到“同一个文件”上，而是会继续写到新的带时间戳目录里。所以刚跑完一轮实验却一时没看到，通常不是结果丢了，而是保存在了新的时间戳目录中。

例如从仓库根目录启动：

- `results/sokoban/TwoPlayer-Sokoban-v0/happo/trial_happo/seed-00001-2026-06-20-17-35-26/`

这个目录里通常重点看：

- `config.json`
  本次训练参数配置
- `progress.txt`
  简单文本进度
- `logs/summary.json`
  主要标量日志
- `models/`
  保存下来的模型参数

### 6.2 现有实验脚本分组

仓库根目录保留早期单实验脚本，例如 `run6_0.sh`、`run6.1.sh`、`run6.2.sh`、`run7*.sh`。新增大批脚本已经放入子目录，避免根目录过于拥挤：

- `run0/`
  原始奖励 baseline。`reward_finished=10`，无任何 shaping。包含 20M 独训和历史伪混合。
- `run2/`
  早期 HAPPO 超参 sweep，如 actor/critic 学习率、PPO epoch、clip、hidden size。
- `run7/`
  reward shaping v2 实验，包含 reverse-push distance、useful-position distance、unreachable-distance cap 和 CNN 对照。
- `run8/`
  记录朴素 BFS shaping 已筛选方案的继续训练和混合训练。每个方案先从已经跑过一轮的 `resume5m` 结果继续两轮，每轮 `5e6` steps；随后各自独立进行 v0/v1/v2 的 1:1:1 混合训练四轮。
- `run9/`
  记录 GRU/RNN + 动作历史 observation 的新实验。从 0 开始训练，不加载旧 MLP checkpoint。
- `run10/`
  v2 cap12 长程续训和历史伪混合。
- `run11/`
  GRU no-history / one-action-history 的补齐实验，尤其用于补过去没跑完的 7x7 伪混合阶段。
- `run12/`
  true-mix padded smoke tests，用来验证不固定 `dim_room/num_boxes` 时的 v0/v1/v2 混合链路。
- `run13/`
  7x7 到 13x13 的蒸馏尝试。实验结果不理想，保留为失败尝试记录。
- `run14/`
  13x13 random padding 从头训练 + true mix。
- `run15/`
  13x13 top-left padding curriculum + true mix，减少 random padding 带来的平移泛化负担。
- `run16/`
  actor-side credit assignment 小型算法实验。默认不影响旧代码，只有显式传入 `--actor_credit_coef > 0` 才启用。

每个 `run*/` 文件夹都放了一个 `README.md`，优先看对应 README 再决定是否运行脚本。

`run8/` 中的总控脚本：

- `run8/run_general8.1.sh`: strongpot 完整流水线
- `run8/run_general8.2.sh`: nodeadlock 完整流水线
- `run8/run_general8.3.sh`: nodl-nopush 完整流水线
- `run8/run_general8.4.sh`: noshape baseline 完整流水线
- `run8/run_general8.5.sh`: 依次跑上面四套

`run9/` 中的主要脚本：

- `run9/run9.1_smoke.sh`: GRU 小规模 smoke test
- `run9/run9.2_bfs_strongpot_rnn.sh`: 旧 BFS strongpot + GRU
- `run9/run9.3_bfs_nodl_nopush_rnn.sh`: 旧 BFS nodl-nopush + GRU
- `run9/run9.4_v2_cap12_rnn.sh`: v2 cap12 + GRU
- `run9/run_general9.1.sh`: 依次跑三条主 RNN 实验

`run16/` 的 credit-assignment 对照：

- `run16/run16.1_v1_nodl_nopush_credit_progress005.sh`: `active_progress`，系数 `0.05`
- `run16/run16.2_v1_nodl_nopush_credit_progress010.sh`: `active_progress`，系数 `0.10`
- `run16/run16.3_v1_nodl_nopush_credit_reward002.sh`: `active_reward`，系数 `0.02`
- `run16/run_general16_credit_assignment.sh`: 依次跑上面三条

这些脚本内部都会自动切回仓库根目录，因此可以从仓库根目录运行：

```bash
bash run8/run_general8.1.sh
bash run9/run9.1_smoke.sh
bash run16/run_general16_credit_assignment.sh
```

### 6.3 现有可视化方式

#### 方式 A：直接画图

已经新增脚本：

- `HARL/scripts/plot_sokoban_metrics.py`

用法示例：

```powershell
python HARL/scripts/plot_sokoban_metrics.py --run-dir results/happo-shaped/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_shaped/<run-dir>
```

会在该实验目录下生成：

- `plots/reward_curves.png`
- `plots/optimization_curves.png`
- `plots/sokoban_curves.png`（当日志里存在这些指标时）
- `plots/shaping_curves.png`（当日志里存在 shaping 指标时）
- `plots/metric_index.md`

该脚本优先读取 `logs/summary.json`；如果该文件不存在，也会递归读取 `logs/**/events.out.tfevents*`，因此不依赖 TensorBoard UI 也能画出大部分曲线。

#### 方式 B：TensorBoard

如果安装了 `tensorboard`，也可以直接看事件文件：

```bash
PYTHONNOUSERSITE=1 python -m tensorboard.main \
  --logdir ./results \
  --host 0.0.0.0 \
  --port 6006 \
  --load_fast=false
```

注意：

- 训练如果是从仓库根目录启动，就看 `./results`
- 如果是从 `HARL/` 目录启动，就看 `HARL/results`
- `tensorboard==2.13.0` 需要和 `protobuf<5` 搭配，否则 HParams 插件会报错
- 它还会直接导入 `pkg_resources`，因此这里固定 `setuptools==65.5.0`；新版 `setuptools` 已移除该模块
- TensorBoard 2.13 仍使用 `np.string_`，因此需要 `numpy<2`，这里固定为 `numpy==1.26.4`
- setup 脚本现在会一并固定 `numpy==1.26.4`、`tensorboard==2.13.0`、`tensorboardX==2.6.5`、`protobuf<5`、`setuptools==65.5.0`、`six`

### 6.4 目前建议重点监测哪些指标

结合 HARL 当前实现，以及 HAPPO / PPO 类论文里常见的实验图，建议重点看：

- `train_episode_rewards/aver_rewards`
  训练回报
- `eval_average_episode_rewards`
  评估回报，比训练回报更重要
- `critic/average_step_rewards`
  平均 step reward，可近似看“单步表现是否离开纯惩罚区”
- `critic/value_loss`
  critic 是否稳定
- `agent0/ratio`, `agent1/ratio`
  PPO 重要性采样比值，过大或剧烈抖动说明更新不稳
- `agent0/dist_entropy`, `agent1/dist_entropy`
  探索程度，过早塌缩通常不是好信号
- `agent0/actor_grad_norm`, `agent1/actor_grad_norm`, `critic/critic_grad_norm`
  梯度是否爆炸

Sokoban 额外值得监测：

- `sokoban/train_success_rate`
- `sokoban/eval_success_rate`
- `sokoban/train_box_completion_ratio`
- `sokoban/eval_box_completion_ratio`
- `sokoban/train_box_pushes`
- `sokoban/train_conflict_rate`
- `sokoban/train_invalid_action_rate`
- `sokoban/train_noop_rate`
- `sokoban/train_mean_base_reward`
- `sokoban/train_mean_shaping_reward`
- `sokoban/train_mean_distance_shaping_reward`
- `sokoban/train_mean_pushability_shaping_reward`
- `sokoban/train_mean_deadlock_shaping_reward`
- `sokoban/train_mean_agent_box_distance_shaping_reward`

说明：

- 在当前默认的 `turn_based` 控制方式下，`conflict_rate` 理论上应接近 `0`
- 如果 `invalid_action_rate` 升高，说明非激活 agent 仍频繁尝试输出非 `noop`
- 这些 Sokoban 专属指标目前主要在 HAPPO 这条 on-policy 日志链路里会更完整
- HASAC 目前已经有回报、评估长度、平均 step reward 等基础曲线
- `shaping_reward` 如果长期绝对值远大于 `base_reward`，说明辅助奖励开始主导任务，需重新缩放权重

---

## 7. 如何根据结果判断训练是否有效

这一部分是最实用的判断标准。

### 7.1 先看“有没有脱离随机/失败状态”

在当前奖励下，如果策略几乎一直瞎走，常见现象是：

- `eval_average_episode_rewards` 长时间接近 `-20`
- `critic/average_step_rewards` 长时间接近 `-0.1`
- `success_rate` 接近 `0`
- `box_completion_ratio` 长时间很低

这通常意味着：

- 几乎没有把箱子稳定推到目标点
- 大部分 episode 只是消耗步数到结束

### 7.2 真正有效时通常会出现什么

一个更像“开始学到东西了”的信号，通常是这些现象同时出现：

- `eval_average_episode_rewards` 明显上升
- `eval_success_rate` 从 0 开始抬头
- `box_completion_ratio` 上升
- `critic/average_step_rewards` 逐步高于 `-0.1`
- 若策略开始有效率，`eval_average_episode_length` 可能下降

要注意：

- “episode 变短”不一定是好事
- 如果 episode 变短但 reward 仍很差，可能只是更快失败

### 7.3 HAPPO 训练是否稳定

HAPPO 是 PPO 系列，稳定性还要看优化指标：

- `ratio` 最好围绕 1 附近波动，不要经常飘得很远
- `dist_entropy` 可以逐步下降，但不要一开始就迅速塌到很低
- `value_loss` 可以波动，但不应长期爆炸
- `grad_norm` 不应频繁出现极端大值或 NaN

如果出现下面情况，通常说明训练有问题：

- reward 不升，entropy 很快塌缩
- reward 不升，ratio 剧烈波动
- value loss 经常大幅爆炸
- 不同 seed 之间差异极端大

### 7.4 最后一定要看评估，不只看训练

训练曲线常常更乐观，真正该重点看的是：

- `eval_average_episode_rewards`
- `eval_success_rate`
- 不同 seed 的平均表现

比较推荐的做法是：

1. 先跑短实验，确认链路通
2. 再拉长训练步数
3. 至少跑 3 个 seed
4. 看均值和波动，而不是只看单条最好看的曲线

---

## 8. 整体结构梳理：网络、输入、处理、输出、动作

下面把整个“从 observation 到最终动作”的链路理顺。

### 8.1 HAPPO：几张网络、各自做什么

默认配置 `share_param: False`，所以当前是：

- 2 个 actor 网络
  agent0 一个，agent1 一个
- 1 个中心化 V critic

对应文件：

- actor 外层算法：`HARL/harl/algorithms/actors/happo.py`
- actor 网络主体：`HARL/harl/models/policy_models/stochastic_policy.py`
- 动作头：`HARL/harl/models/base/act.py`
- critic 外层算法：`HARL/harl/algorithms/critics/v_critic.py`
- critic 网络主体：`HARL/harl/models/value_function_models/v_net.py`

#### HAPPO actor 的输入

每个 agent 都吃自己的局部 observation。

当前 observation 在 `HARL/harl/envs/sokoban/sokoban_env.py` 里构造，主要包括：

- 墙体位置
- 目标点位置
- 箱子在目标上的位置
- 箱子不在目标上的位置
- 自己的位置
- 另一名玩家的位置
- 当前优先执行权
- 当前步数占最大步数的比例
- 可选动作历史：过去 `action_history_len` 步双方实际执行动作的 one-hot，默认关闭

这些通道会被展平，形成一个 1D 向量。当前 TwoPlayer-Sokoban-v0 下默认维度是 `297`。如果设置 `--action_history_len 8`，会额外加入 `8 * 2 * 9 = 144` 维动作历史，因此 vector observation 变为 `441` 维。

#### HAPPO actor 的处理过程

处理链路是：

1. observation 输入 `MLPBase`
2. 经过若干隐藏层
3. 如果开启 RNN，再经过 `RNNLayer`
4. 最后进入 `ACTLayer`

当前默认 `happo.yaml` 中：

- `hidden_sizes: [128, 128]`
- `use_recurrent_policy: False`

所以当前默认实际上是纯 MLP，不带 RNN。

如果在命令行开启：

```bash
--use_recurrent_policy True
--recurrent_n 1
--data_chunk_length 25
```

HARL 会复用内置 `RNNLayer`，其底层是 GRU。`run9/` 脚本同时开启 `--action_history_len 8`，让每个 agent 能显式看到过去 8 步双方动作；GRU hidden state 则可继续积累更长历史。由于 RNN 和动作历史都会改变模型结构或输入维度，`run9/` 默认从 0 训练，不从旧 MLP checkpoint resume。

#### HAPPO actor 的输出

输出是离散动作分布，对应 9 个动作：

- `0`: noop
- `1~4`: push up/down/left/right
- `5~8`: move up/down/left/right

训练时 actor 会输出：

- 采样动作
- 该动作的 log prob

评估时一般取 deterministic/mode 动作。

#### HAPPO critic 的输入与输出

critic 吃的是共享状态 `share_obs`，当前也是展平后的 297 维向量。

处理链路：

1. `share_obs` 输入 `VNet`
2. 经过 `MLPBase`
3. 如开启 RNN 则过 `RNNLayer`
4. 线性层输出一个标量 `V(s)`

这个值用于：

- GAE
- return 计算
- actor 优势估计

#### HAPPO 最终如何变成环境动作

流程是：

1. agent0 actor 产出一个 0~8 动作
2. agent1 actor 产出一个 0~8 动作
3. `sokoban_env.py` 收到这两个动作
4. 默认 `turn_based` 模式下，只允许当前轮到的 agent 动作，另一位 agent 只能输出 `noop`
5. 若当前轮到的是 agent1，则动作会映射到底层双人 Sokoban 的 `9~16`
6. 底层环境真正 step 一次，然后轮换到下一位 agent

也就是说：

- “策略仍然各自输出动作分布”
- “但默认语义已经改成严格轮流行动，而不是同一步竞争执行权”

### 8.2 HASAC：几张网络、各自做什么

默认 `share_param: False`，当前是：

- 2 个 actor 网络
- 2 个 Q critic 网络
- 2 个 target Q critic 网络

也就是典型 SAC 的 twin-Q 结构，只不过扩展成了多智能体版本。

对应文件：

- actor 外层算法：`HARL/harl/algorithms/actors/hasac.py`
- actor 网络主体：`HARL/harl/models/policy_models/stochastic_mlp_policy.py`
- critic 外层算法：`HARL/harl/algorithms/critics/soft_twin_continuous_q_critic.py`
- twin critic 基类：`HARL/harl/algorithms/critics/twin_continuous_q_critic.py`
- critic 网络主体：`HARL/harl/models/value_function_models/continuous_q_net.py`

#### HASAC actor 的输入

和 HAPPO 一样，每个 actor 吃自己 agent 的局部 observation，也是当前 297 维向量。

#### HASAC actor 的处理过程

当前默认 `hasac.yaml` 中：

- `hidden_sizes: [256, 256]`

处理链路：

1. observation 输入 `MLPBase`
2. 得到 actor feature
3. 进入 `ACTLayer` 或 logits 头

对于离散动作空间，HASAC 在训练 critic 时会走：

- `get_logits`
- `gumbel_softmax`

这样得到 one-hot 风格动作，便于和 joint action 一起喂给 Q 网络。

#### HASAC actor 的输出

有两种“形态”：

- 与环境交互时：输出离散动作 id
- 给 critic 训练时：输出 one-hot / 近似 one-hot 动作表示，以及对应 log prob

#### HASAC critic 的输入

Q 网络吃的是：

- 共享状态 `share_obs`
- 所有 agent 的 joint action

其中 joint action 会被拼接起来：

- 如果是离散动作，会先转 one-hot
- 两个 agent 的 one-hot 再拼起来

#### HASAC critic 的处理过程

处理链路：

1. `share_obs` 输入 `ContinuousQNet`
2. 与 joint action 拼接
3. 输入 MLP
4. 输出一个标量 `Q(s, a_joint)`

因为是 twin-Q，所以有两套：

- `critic`
- `critic2`

并各自有：

- `target_critic`
- `target_critic2`

训练时取两者较小值，减少 Q 过估计。

#### HASAC 最终如何变成环境动作

环境交互阶段：

1. agent0 actor 产出动作
2. agent1 actor 产出动作
3. `sokoban_env.py` 做冲突消解
4. 实际执行一个底层动作

训练阶段：

1. replay buffer 采样 batch
2. actor 重新给 next state 产出 next action
3. twin Q 计算 soft target
4. 更新 critic
5. 再按 agent 顺序更新 actor

---

## 9. 当前日志里新增了什么

除了 HARL 原本就有的 reward / loss / entropy / grad norm 之外，这次还额外补了更适合 Sokoban 的统计：

- 成功率
- 箱子完成比例
- 每局推箱次数
- 每局移动次数
- noop 比例
- 冲突比例
- agent0 / agent1 实际被执行的比例

这些更适合回答下面这种问题：

- 策略是真的开始协作了吗？
- 还是只是在随机乱动？
- reward 没涨是因为不敢动，还是因为冲突太多？

---

## 10. 推荐阅读顺序

如果组员是第一次接触这个项目，建议按下面顺序看：

1. `README.md`
2. `HARL/harl/envs/sokoban/sokoban_env.py`
3. `HARL/harl/configs/envs_cfgs/sokoban.yaml`
4. `HARL/examples/train.py`
5. `HARL/harl/runners/on_policy_ha_runner.py`
6. `HARL/harl/algorithms/actors/happo.py`
7. `HARL/harl/runners/off_policy_ha_runner.py`
8. `HARL/harl/algorithms/actors/hasac.py`
9. `HARL/harl/algorithms/critics/soft_twin_continuous_q_critic.py`
10. `gym-sokoban/gym_sokoban/envs/sokoban_env.py`
11. `gym-sokoban/gym_sokoban/envs/sokoban_env_two_player.py`

这样最容易建立“底层环境 -> HARL 封装 -> 算法训练”的整体理解。

---

## 11. 后续可以继续做什么

后续可以继续沿当前框架做：

- 换更高版本地图或不同房间大小
- 修改 observation 设计
- 修改冲突消解规则
- 设计 reward shaping
- 比较 HAPPO 与 HASAC 的样本效率和最终性能
- 跑多 seed、多 GPU、大步数实验
- 在此基础上加入自己的创新点
