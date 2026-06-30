# Final GRU0 Evaluation Metrics

本文档记录最终提交模型 `GRU0` 的 7x7 evaluation 指标，供报告“最优模型”章节填写时参考。

## 模型版本

- 模型名称：`GRU0`
- 含义：GRU encoder，不显式输入动作历史。
- 主线设置：7x7 Sokoban，HAPPO，`v1-nodl-nopush` BFS reward shaping。
- 链条来源：
  - `happo-rnn-bfs-nodl-nopush-nohist`
  - `happo-rnn-bfs-nodl-nopush-nohist-resume10m`
  - `happo-rnn-bfs-nodl-nopush-nohist-resume15m`
  - `happo-rnn-bfs-nodl-nopush-nohist-resume20m`
  - `happo-rnn-bfs-nodl-nopush-nohist-mixv012-r1`
  - `happo-rnn-bfs-nodl-nopush-nohist-mixv012-r2`
  - `happo-rnn-bfs-nodl-nopush-nohist-mixv012-r3`
  - `happo-rnn-bfs-nodl-nopush-nohist-mixv012-r4`

## 最终 Eval 指标口径

最后一个 logged eval 点位于 stitched step `39.8M`。由于 eval 不是在精确 `40.0M` 处记录，因此报告中可写作“约 40M steps 后”。

需要注意：图中的曲线通常经过 smoothing，因此视觉上可能低于最后一个 raw eval 点；下面表格中的 final raw 数值是直接从 TensorBoard scalar/event 记录读取的未平滑结果。

| metric | value |
| --- | ---: |
| final raw `sokoban/eval_success_rate` | `1.0000` |
| final raw `sokoban/eval_box_completion_ratio` | `1.0000` |
| last-5 mean `sokoban/eval_success_rate` | `0.6851` |
| last-5 mean `sokoban/eval_box_completion_ratio` | `0.8014` |
| last-10 mean `sokoban/eval_success_rate` | `0.7159` |
| last-10 mean `sokoban/eval_box_completion_ratio` | `0.8080` |
| last-20 mean `sokoban/eval_success_rate` | `0.6074` |
| last-20 mean `sokoban/eval_box_completion_ratio` | `0.7325` |

## 说明

- 两个指标均来自 TensorBoard scalar/event 记录，而不是手工估计。
- final raw `eval_success_rate=1.0000` 表示最后一次 evaluation 中成功率为 100%。
- final raw `eval_box_completion_ratio=1.0000` 表示最后一次 evaluation 中箱子到位比例为 100%。
- 图像中看起来低于 `1.0` 的原因是曲线经过平滑，同时末段 eval 本身存在一定波动；最近 10 次 evaluation 的平均 success rate 约为 `0.716`，平均 box completion ratio 约为 `0.808`。
- 报告中可以如实写作：“最终 GRU0 在最后一次 evaluation 中达到 100% success rate 和 100% box completion ratio。图中曲线为平滑后的结果，因此视觉上低于最后一次 raw evaluation 点。”
