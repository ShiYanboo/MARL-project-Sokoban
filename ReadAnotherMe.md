- 请直接告诉我HAPPO和HASAc默认的最优附近超参？结合这个任务推断一下大致的范围？环境给的奖励是否合理，按照初始奖励能训练出来吗？



- 没有一个“默认最优超参”能直接保证 Sokoban 最好收敛，尤其这是一个双智能体、稀疏奖励、离散动作、协作推箱子的任务。不过结合 HARL 默认配置和你们现在这个环境形态，我会这样判断。

HAPPO 这边，默认配置其实已经在一个“能作为起点”的区间里了。对当前 Sokoban，我建议先把它当主力基线。比较稳的范围大概是：`lr = 5e-5 ~ 2e-4`，`critic_lr = 2e-4 ~ 5e-4`，也就是 critic 不要比 actor 更慢；`clip_param = 0.1 ~ 0.2`；`entropy_coef = 0.005 ~ 0.01`；`ppo_epoch = 5 ~ 10`；`gamma = 0.99`、`gae_lambda = 0.95` 基本不用先动。本轮实验保持 `run1.sh` 的 MLP 宽度 `[256,256]`，先别把网络结构和训练稳定性问题混在一起。训练规模上，`num_env_steps` 至少要看 `1e6` 量级才开始有判断意义，难一点的设置经常要到 `3e6 ~ 1e7`。本机可以先 `8~16` 个 rollout threads，服务器上可以上到 `16~64`。

HASAC 这边我不建议一开始就指望它比 HAPPO 更稳。它在这种稀疏协作离散任务上通常更挑 reward 和采样分布。比较合理的起始范围是：`lr/critic_lr = 1e-4 ~ 5e-4`，常用 `3e-4`；`batch_size = 256 ~ 512`，我反而不太建议一上来用默认 `1000`；`buffer_size = 1e5 ~ 1e6` 都可以；`warmup_steps = 5e3 ~ 2e4`；`train_interval = 25 ~ 100`；`polyak = 0.005` 保持默认即可；`n_step` 我会建议先试 `3 ~ 10`，比默认 `20` 更适合现在这个任务。温度参数 `alpha` 如果固定，先试 `0.005 ~ 0.05`，但更建议直接开 `auto_alpha=True`，因为当前任务探索压力比较大。

如果只给一句“最可能接近可用”的建议，我会这么配：
- HAPPO：`lr=1e-4`，`critic_lr=3e-4`，`entropy_coef=0.005 ~ 0.01`，`clip_param=0.1`，`ppo_epoch=5`，`hidden_sizes=[256,256]`，`num_env_steps>=3e6`
- HASAC：`lr=3e-4`，`critic_lr=3e-4`，`auto_alpha=True`，`batch_size=256`，`warmup_steps=10000`，`train_interval=50`，`n_step=5`，`hidden_sizes=[256,256]`

关于奖励，我的判断是：现在这套奖励“语义上合理”，但“学习上偏难”。你们当前底层 Sokoban 奖励本质上是：
- 每步 `-0.1`
- 箱子推上目标 `+1`
- 箱子离开目标 `-1`
- 全部完成 `+10`

这套奖励本身没错，属于经典 Sokoban 风格；问题是它偏稀疏，而且你们现在是双人协作、共享 team reward，所以 credit assignment 仍然比较难。现在默认控制方式已经改成 `turn_based`，不再让两个 agent 在同一步里竞争执行权；这比原先的冲突裁决更适合作为 HAPPO 基线。结论不是“奖励错了”，而是“原始奖励可以训练，但大概率比较慢、比较不稳，而且很吃地图难度和训练步数”。如果直接用当前初始奖励在默认难度上硬训，我认为：
- HAPPO 有机会训出来，但需要更长 steps，且最好先从简单地图开始
- HASAC 更可能前期卡住，尤其如果不做 reward shaping 或 curriculum

更实际的建议是先把环境难度降下来验证可学性，比如：
- `num_boxes=1` 或 `2`
- `dim_room` 先小一点，比如 `7` 或 `8`
- `max_steps=80 ~ 120`

如果这样都学不动，那就不是超参问题了，而更像是 credit 或观测/动作机制的问题；如果简单版能学动，再慢慢加难度，就说明原始奖励至少是“可用的”。

一句话总结：当前任务上，HAPPO 比 HASAC 更像主线方案；原始奖励合理但偏稀疏，理论上能训，实践上建议配合更简单的起始关卡或额外 shaping，不然很容易长期停在“平均每步接近 -0.1、总回报很差”的状态。

---

## 服务器最小训练 Instructions

这一节只保留“第一次上 Linux 服务器就能开训”的最小步骤。

### 1. 第一次配置环境

默认约定：

- 仓库根目录：`/path/to/MARL-project-sokoban`
- `HARL/` 和 `gym-sokoban/` 位于同一级
- conda 环境名默认：`harl-sokoban`

第一次进入服务器后执行：

```bash
cd /path/to/MARL-project-sokoban
bash HARL/scripts/setup_sokoban_linux.sh
conda activate harl-sokoban
python HARL/scripts/verify_sokoban_setup.py
```

如果服务器 CUDA 版本不同，可以显式指定环境名和 PyTorch CUDA 轮子通道：

```bash
cd /path/to/MARL-project-sokoban
ENV_NAME=harl-sokoban-cu121 TORCH_CHANNEL=cu121 bash HARL/scripts/setup_sokoban_linux.sh
conda activate harl-sokoban-cu121
python HARL/scripts/verify_sokoban_setup.py
```

当前 setup 脚本支持：

- `TORCH_CHANNEL=cu118`
- `TORCH_CHANNEL=cu121`
- `TORCH_CHANNEL=cu124`
- `TORCH_CHANNEL=cu126`
- `TORCH_CHANNEL=cpu`

如果只想确认 GPU 是否已被 PyTorch 识别：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### 2. 开始训练

建议先从 HAPPO 作为主 baseline 开始。

HAPPO 最小训练命令：

```bash
cd /path/to/MARL-project-sokoban
conda activate harl-sokoban
python HARL/examples/train.py \
  --algo happo \
  --env sokoban \
  --scenario TwoPlayer-Sokoban-v0 \
  --exp_name server_happo \
  --cuda True \
  --control_mode turn_based \
  --n_rollout_threads 16 \
  --n_eval_rollout_threads 4 \
  --episode_length 80 \
  --num_env_steps 3000000 \
  --max_steps 80 \
  --dim_room 7 \
  --num_boxes 1 \
  --lr 1e-4 \
  --critic_lr 3e-4 \
  --entropy_coef 0.01 \
  --ppo_epoch 5 \
  --clip_param 0.1 \
  --hidden_sizes "[256, 256]" \
  --log_interval 5 \
  --eval_interval 25 \
  --eval_episodes 20 \
  --use_linear_lr_decay True \
  --use_eval True
```

HASAC 最小训练命令：

```bash
cd /path/to/MARL-project-sokoban
conda activate harl-sokoban
python HARL/examples/train.py \
  --algo hasac \
  --env sokoban \
  --scenario TwoPlayer-Sokoban-v0 \
  --exp_name server_hasac \
  --cuda True \
  --n_rollout_threads 16 \
  --n_eval_rollout_threads 2 \
  --num_env_steps 3000000 \
  --lr 3e-4 \
  --critic_lr 3e-4 \
  --auto_alpha True \
  --batch_size 256 \
  --buffer_size 100000 \
  --warmup_steps 10000 \
  --train_interval 50 \
  --n_step 5 \
  --hidden_sizes "[256, 256]" \
  --eval_interval 5000 \
  --eval_episodes 10 \
  --use_eval True
```

如果想先跑一个更容易训通的版本，可以额外补环境参数：

```bash
--max_steps 100 --dim_room 7 --num_boxes 1
```

仓库根目录还提供了一组现成脚本：

- `run2.sh`：`happo-base`，turn-based，`7x7, 2 boxes`
- `run3.sh`：`happo-medium`，turn-based，`7x7, 3 boxes`
- `run2_21.sh`：`happo-cnn-base`，使用随机初始化的三层 CNN encoder
- `run5.sh`：`happo-cnn-shaped`，通过 `--use_reward_shaping True` 在 CNN baseline 上增加 BFS 距离、pushability、死锁和 agent-box 距离 shaping，并通过 `--reward_finished 20` 提高整局完成奖励；其他旧脚本仍使用原始的 `+10`
- `run6.sh`：`happo-shaped`，回到 MLP `[512,512]`，并使用 `deadlock_penalty_mode increase` 避免同一死锁状态每步重复重罚
- `run6_0.sh`：`happo-shaped-strongpot`，在 `run6.sh` 基础上把正向势函数放大到 `distance_shaping_weight=0.20`、`agent_box_distance_shaping_weight=0.10`
- `run6.1.sh`：去掉 deadlock，保留较大的 box-target 和 agent-box 正向势函数
- `run6.2.sh`：去掉 deadlock 和 pushability，保留较大的 box-target 和 agent-box 正向势函数
- `run6.3.sh`：去掉 deadlock、pushability 和 box-target，只保留较大的 agent-box 势函数
- `run6.4.sh`：全部 shaping 关闭，保留 `reward_finished 20`，作为无 shaping 对照

当前建议优先比较 `run6_0.sh`、`run6.1.sh`、`run6.2.sh` 和 `run6.4.sh`。如果 `run6_0.sh` 比无 shaping 明显更快脱离随机区间，说明正向势函数有帮助；如果 `run6.1/run6.2` 明显更稳，说明 deadlock 或 pushability 的负向项仍偏强。

后续大量续训脚本已经按系列放入子目录：

- `run8/`：朴素 BFS shaping 已筛选方案的继续训练与 v0/v1/v2 混合训练
- `run9/`：GRU/RNN + 动作历史 observation 的实验

`run8/` 的设计是“每个被筛出来还不错的奖励方案，各自完整跑自己的流水线”，而不是所有方案混在同一个 prefix 下：

- `run8/run_general8.1.sh`：`strongpot`，先从 `happo-shaped-strongpot-resume5m` 继续两轮，每轮 `5e6` steps，再做 v0/v1/v2 的 1:1:1 混合训练四轮
- `run8/run_general8.2.sh`：`nodeadlock`，同样两轮继续训练 + 四轮混合训练
- `run8/run_general8.3.sh`：`nodl-nopush`，同样两轮继续训练 + 四轮混合训练
- `run8/run_general8.4.sh`：`noshape`，无 shaping baseline，同样两轮继续训练 + 四轮混合训练
- `run8/run_general8.5.sh`：依次跑上面四套完整流水线

所有 `run8/` 脚本默认使用 GPU5，并且显式传入旧 BFS 兼容模式：

```bash
--box_target_distance_mode bfs
--agent_box_distance_mode adjacent
```

因此它们继续训练早期朴素 BFS 模型时，不会悄悄切换到 v2 的 reverse-push/useful-position reward 定义。

`run9/` 的 GRU 脚本不从旧 MLP 模型 resume，因为 `--use_recurrent_policy True` 和 `--action_history_len 8` 都会改变网络结构或输入维度，严格加载旧 checkpoint 会 shape mismatch。可先跑：

```bash
bash run9/run9.1_smoke.sh
```

确认链路没问题后，再跑：

```bash
bash run9/run_general9.1.sh
```

### 2.1 reward shaping v2：有效推箱距离

原来的 `distance_shaping_weight` 使用箱子到目标的普通 BFS 地面距离。这个信号虽然平滑，但没有考虑 Sokoban 的核心约束：箱子只能被推，不能自己走过去；箱子前进方向的前后两格都需要可用，玩家还需要能站到箱子身后。因此普通 BFS 会奖励一些几何上靠近目标、但实际上无法沿该方向推动的状态。

当前 v2 把箱子-目标距离改成了静态 reverse-push distance table。对每个目标预计算一张表：

```text
target_push_dist[target_id][r][c]
= 如果箱子在 (r, c)，静态地图上最少推多少次能到 target_id。
```

预计算伪代码如下：

```python
for target in targets:
    dist = full_grid(INF)
    dist[target] = 0
    queue = deque([target])

    while queue:
        cur = queue.popleft()
        for d in directions:
            # 正向推箱是 prev -> cur
            prev = cur - d
            player_stand = prev - d

            if inside(prev) and inside(player_stand):
                if not wall(prev) and not wall(player_stand):
                    if dist[prev] > dist[cur] + 1:
                        dist[prev] = dist[cur] + 1
                        queue.append(prev)
```

这样得到的是“有效 push 步数”而不是地面距离。多箱子多目标时不会贪心选择最近目标，而是对每个箱子、每个目标查表组成 cost matrix，再继续使用最小匹配 DP。复杂度为：

- 距离表预计算：`O(T * H * W)`，`T` 是目标数量。
- 每步构造 cost：`O(B * T)`，`B` 是箱子数量。
- matching DP：`O(B * T * 2^T)`；当前 Sokoban 箱子数量较少，这部分开销可接受。

需要注意的是，这张表只使用静态墙和目标，不把其他箱子当成长程障碍。这样做是有意的：它让 reward 保持平滑，不会因为另一个箱子暂时挡路就把距离突然变成不可达。但它仍然只是启发式 shaping，不是完整 Sokoban solver。

玩家-箱子距离也同步改成了“玩家到有效站位距离”。对当前每个箱子和每个方向，代码会临时检查一次真实局面下是否可推：`front` 和 `rear` 在地图内且不是墙，`front` 没有箱子或玩家，`rear` 没有箱子，并且至少一个玩家能走到 `rear`。如果把箱子推到 `front` 后，全局有效匹配距离下降，则 `rear` 被认为是有效站位。`agent_box_distance_shaping_weight` 现在奖励玩家靠近这些有效站位，而不是靠近箱子任意相邻格。

此外新增了一个默认关闭的有效推动作奖励：

```bash
--useful_push_shaping_weight 0.0
```

当一步 transition 恰好移动了一个箱子，并且全局有效箱子-目标距离下降时，`useful_push_applied=1`。实际奖励为：

```text
useful_push_shaping_weight * useful_push_applied
```

默认权重是 0，因此所有旧脚本数值行为保持兼容；但在开启 reward shaping 时，日志里会额外记录 `useful_push_shaping_reward`、`useful_push_applied` 和 `useful_push_distance_delta`，方便后续分析 agent 是否真的学会了“朝有效方向推箱子”。

当前代码为了兼容旧实验，把 reward shaping 距离语义做成了命令行可切换：

```bash
--box_target_distance_mode reverse_push
--agent_box_distance_mode useful
```

这是 v2 默认含义：box-target 用 reverse-push distance，agent-box 用有效站位距离。

```bash
--box_target_distance_mode bfs
--agent_box_distance_mode adjacent
```

这是早期朴素 BFS 含义：box-target 只考虑墙，不考虑其他箱子和 agent；agent-box 会把箱子和另一名 agent 当作障碍，计算玩家到箱子相邻格的 BFS 距离并加 1。`run8/` 续训脚本使用这一组参数复现实验早期的 reward 定义。

还有一个诊断参数：

```bash
--shaping_unreachable_distance 12
```

它会把不可达距离裁剪为指定值。默认不裁剪，仍使用地图面积作为不可达距离。这个参数主要用于 v2 奖励尺度诊断，避免不可达状态从 `-49` 这类较大差分突然主导训练。

### 2.2 GRU/RNN 与动作历史

HARL 本身已经有 GRU recurrent policy/critic，代码位置：

- `HARL/harl/models/base/rnn.py`
- `HARL/harl/models/policy_models/stochastic_policy.py`
- `HARL/harl/models/value_function_models/v_net.py`

Sokoban 环境侧新增了：

```bash
--action_history_len 8
```

它会把过去 8 步双方实际执行动作加入 observation。每一步记录两个 agent 的 one-hot 动作；turn-based 下 active agent 记录真实执行动作，inactive agent 记录 noop。vector observation 因此会从默认 `297` 维变成：

```text
297 + 8 * 2 * 9 = 441
```

RNN 脚本同时打开：

```bash
--use_recurrent_policy True
--recurrent_n 1
--data_chunk_length 25
```

`data_chunk_length` 要能整除 `episode_length`，当前脚本 `episode_length=150`、`data_chunk_length=25` 是匹配的。由于 RNN 和动作历史改变了网络结构/输入维度，`run9/` 默认从头训练，不从旧 MLP checkpoint resume。

### 3. 命令行改参数

HARL 支持直接在训练命令后覆盖 yaml 默认值。比如：

```bash
python HARL/examples/train.py \
  --algo happo \
  --env sokoban \
  --scenario TwoPlayer-Sokoban-v0 \
  --exp_name ablation_lr \
  --run_name_prefix happo-ablation-lr \
  --lr 1e-4 \
  --critic_lr 1e-4 \
  --entropy_coef 0.01 \
  --num_env_steps 1000000
```

环境相关常改项：

- `--max_steps`
- `--dim_room`
- `--num_boxes`
- `--num_gen_steps`

`--exp_name` 决定实验目录；`--run_name_prefix` 同时决定 `results/` 下的分组目录和最后一级可读目录名。设置后会自动加入训练步数、episode length、CUDA 设备、actor/critic 学习率、PPO epoch、clip、评估局数、seed 和时间戳。例如：

```text
results/happo-ablation-lr/.../happo-ablation-lr-steps1000000-len150-env0-lr1e-4-vlr1e-4-ppoepoch5-clip0.1-evalepi20-seed1-YYYY-MM-DD-HH-MM-SS
```

若前缀包含 `hasac`，目录名还会加入 `entrocoe<alpha>`；`auto_alpha=True` 时写作 `entrocoeauto`。不传该参数时保持旧的 `seed-xxxxx-时间戳` 命名。

CNN 版本使用 `9 x H x W` 的语义输入通道：墙、目标、箱子到位、普通箱子、自己位置、队友位置、自己是否 active、队友是否 active、episode 进度。它不是自然图像，因此不加载 ImageNet 预训练模型，而是用 `Conv(32)-Conv(64)-Pool-Conv(64)` 从头训练。turn-based 模式下 inactive agent 只能输出 noop，并在 actor buffer 中设置 `active_mask=0`；因此它不参与该步 policy loss。critic 仍使用 centralized state 和 shared team reward。

训练相关常改项：

- `--n_rollout_threads`
- `--num_env_steps`
- `--episode_length`
- `--lr`
- `--critic_lr`
- `--entropy_coef`
- `--ppo_epoch`
- `--batch_size`
- `--warmup_steps`
- `--train_interval`

### 4. 实时可视化

最方便的是 TensorBoard。服务器上不需要直接本地打开网页，远程连接也可以看。

先在服务器启动：

```bash
cd /path/to/MARL-project-sokoban
conda activate harl-sokoban
PYTHONNOUSERSITE=1 python -m tensorboard.main \
  --logdir ./results \
  --host 0.0.0.0 \
  --port 6006 \
  --load_fast=false
```

如果 TensorBoard 报 `including_default_value_fields`，是 `tensorboard` 和 `protobuf` 版本不匹配；如果报 `No module named 'pkg_resources'`，则是新版 `setuptools` 已移除旧 TensorBoard 仍依赖的模块；如果报 `np.string_ was removed`，则是 TensorBoard 2.13 与 NumPy 2 不兼容。当前建议统一固定为：

```bash
python -m pip install --upgrade --force-reinstall \
  "numpy==1.26.4" \
  "setuptools==65.5.0" \
  "tensorboard==2.13.0" \
  "tensorboardX==2.6.5" \
  "protobuf>=3.20,<5" \
  "six>=1.16.0"
```

然后任选一种方式：

- VS Code Remote / SSH：打开 `PORTS` 面板，把服务器 `6006` 端口转发到本地，再点击本地地址
- 纯 SSH：本地运行 `ssh -L 6006:127.0.0.1:6006 user@your_server`，然后浏览器打开 `http://127.0.0.1:6006`

建议重点看这些指标：

- `eval_average_episode_rewards`
- `critic/average_step_rewards`
- `sokoban/eval_success_rate`
- `sokoban/eval_box_completion_ratio`
- `agent0/dist_entropy`
- `agent1/dist_entropy`
- `critic/value_loss`

如果不方便开 TensorBoard，也可以在训练后直接画图：

```bash
cd /path/to/MARL-project-sokoban
python HARL/scripts/plot_sokoban_metrics.py --run-dir results/happo-shaped/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_shaped/<run-dir>
```

`plot_sokoban_metrics.py` 会优先读取 `logs/summary.json`；如果该文件不存在，也会递归读取 `logs/**/events.out.tfevents*`，所以不依赖 TensorBoard UI 也能生成 `reward_curves.png`、`optimization_curves.png`、`sokoban_curves.png` 和 `shaping_curves.png`。

### 5. 结果在哪里看

所有训练结果默认保存在“当前启动目录下的 `results/`”：

```text
results/
```

单次实验目录结构通常是：

```text
results/<env>/<scenario>/<algo>/<exp_name>/<parameter-rich-name-or-seed-timestamp>/
```

重点看这些文件：

- `config.json`：本次实验最终参数
- `progress.txt`：训练进度
- `logs/summary.json` 或 `logs/**/events.out.tfevents*`：主要日志
- `models/`：模型权重
- `plots/`：训练后手动画图生成的图片

注意：每次训练都会新建一个带时间戳的目录，不会覆盖之前的结果。如果刚跑完一轮却没立刻看到，大概率是保存到了新的时间戳目录里。
