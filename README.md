# MARL Sokoban 项目说明

## 1. 当前进度

目前已经完成以下工作：

- 已将 `gym-sokoban` 的双人环境接入 `HARL`
- 已搭好 `HAPPO` 和 `HASAC` 两套训练框架
- 已补好本机验证脚本与 Linux `conda` 安装脚本
- 已完成最小可运行验证，确认训练、评估、保存模型的链路打通
- 已补充结果可视化脚本，可直接把 `summary.json` 画成曲线图

当前重点不是“已经训出很强的策略”，而是“实验框架已经可复现、可扩展、可继续做大规模实验”。

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
- `HARL/harl/envs/sokoban/sokoban_logger.py`
  Sokoban 的 on-policy 日志器，负责额外记录成功率、回合制执行统计、推箱次数等指标
- `HARL/harl/configs/envs_cfgs/sokoban.yaml`
  Sokoban 环境配置

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

当前没有额外重写奖励，仍沿用 `gym-sokoban` 原始奖励。定义位置：

- `gym-sokoban/gym_sokoban/envs/sokoban_env.py`

当前规则是：

- 每步惩罚：`-0.1`
- 箱子被推到目标点：`+1`
- 箱子离开目标点：`-1`
- 所有箱子都到目标点：`+10`

HARL 封装层只做了一件事：把这个 reward 作为团队共享奖励发给两个 agent。

### 5.2 当前 credit 如何分配

环境层面目前没有做手工细粒度 credit shaping，也就是说：

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

- `happo-base-steps10000000-len150-env0-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-YYYY-MM-DD-HH-MM-SS/`

其中 `env0` 来自 `CUDA_VISIBLE_DEVICES=0`。前缀中包含 `hasac` 时，还会追加 HASAC 的温度/熵系数字段，例如 `entrocoe0.005`；开启自动温度时写作 `entrocoeauto`。不传 `--run_name_prefix` 时仍使用原来的 `seed-时间戳` 命名。

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

### 6.2 现有可视化方式

#### 方式 A：直接画图

已经新增脚本：

- `HARL/scripts/plot_sokoban_metrics.py`

用法示例：

```powershell
python HARL\scripts\plot_sokoban_metrics.py --run-dir results\sokoban\TwoPlayer-Sokoban-v0\happo\trial_happo\seed-00001-2026-06-20-17-35-26
```

会在该实验目录下生成：

- `plots/reward_curves.png`
- `plots/optimization_curves.png`
- `plots/sokoban_curves.png`（当日志里存在这些指标时）
- `plots/metric_index.md`

当前我已经为这次 HAPPO 样例运行生成了图：

- `results/sokoban/TwoPlayer-Sokoban-v0/happo/trial_happo/seed-00001-2026-06-20-17-35-26/plots/reward_curves.png`
- `results/sokoban/TwoPlayer-Sokoban-v0/happo/trial_happo/seed-00001-2026-06-20-17-35-26/plots/optimization_curves.png`

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

### 6.3 目前建议重点监测哪些指标

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

说明：

- 在当前默认的 `turn_based` 控制方式下，`conflict_rate` 理论上应接近 `0`
- 如果 `invalid_action_rate` 升高，说明非激活 agent 仍频繁尝试输出非 `noop`
- 这些 Sokoban 专属指标目前主要在 HAPPO 这条 on-policy 日志链路里会更完整
- HASAC 目前已经有回报、评估长度、平均 step reward 等基础曲线

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

这些通道会被展平，形成一个 1D 向量。当前 TwoPlayer-Sokoban-v0 下默认维度是 `297`。

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
