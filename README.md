# 策略搜索与价值学习在 Tetris 上的 2×2 对比

在标准 10×20 的 Tetris 棋盘上，比较两种优化方式和两种函数类的四种组合：

| | 线性 | MLP |
|---|---|---|
| **策略搜索**（交叉熵方法） | 交叉熵 · 线性 | 交叉熵 · MLP |
| **价值学习**（落点后状态的 TD） | 线性 TD | DQN |

四个模型共用同一个环境、同一组 13 个落点后特征（10 列高度、最大高度、空洞数、凹凸度）、同一种动作选择方式（对所有合法落点打分，选最高分），以及同一套评估协议。所有模型都以消行数为训练目标。另外有三个不训练的参考方法：Random、单特征贪心（只看总高度）和 Heuristic（Yiyuan Lee 用遗传算法调出的权重）。

## 结果

每个模型 3 个 seed，每个 seed 最终评估 50 局，评估种子固定且不与训练重叠。每局最多放 5,000 块。

| 方法 | 平均消行 | 各 seed 均值 | 碰到上限的局数 | 训练交互步数 |
|---|---:|---|---:|---:|
| 交叉熵 · MLP | 1,923.6 | 1,842.0 / 1,998.0 / 1,930.8 | 139 / 150 | 约 1,508 万 |
| 交叉熵 · 线性 | 1,838.8 | 1,871.5 / 1,705.4 / 1,939.6 | 121 / 150 | 约 282 万 |
| DQN | 1,781.7 | 1,888.0 / 1,459.7 / 1,997.4 | 123 / 150 | 5 万 |
| Heuristic | 1,453.4 | — | 22 / 50 | — |
| 线性 TD | 10.5 | 14.0 / 15.4 / 2.1 | 0 / 150 | 5 万 |
| 单特征贪心 | 8.9 | — | 0 / 50 | — |
| Random | 0.0 | — | 0 / 50 | — |

碰到上限的局在放满 5,000 块时被截断，所以前四个方法的数值是下限，彼此之间的差距不能据此排出确定的名次。

![学习曲线](results_10x20/figures/learning_curves.png)

![全部 7 个方法](results_10x20/figures/all_methods.png)

更多图表在 [`results_10x20/figures/`](results_10x20/figures/)，完整汇总表见 [`results_10x20/summary.md`](results_10x20/summary.md)。

## 代码结构

```
t2x2/
├── env.py         环境：分组动作、落点后特征、奖励 = 消行数
├── evaluate.py    统一的评估协议
├── policies.py    基线、交叉熵的打分函数、TD 的价值网络
├── train_ce.py    交叉熵方法（线性 / MLP）
├── train_td.py    落点后状态的 TD 学习（线性 TD / DQN）
├── baselines.py   评估三个参考方法
└── plot.py        生成图表和汇总表
tetris_gymnasium/  Tetris 环境库（第三方，见下方说明）
run_10x20.sh       复现全部结果
results_10x20/     各次运行的配置、学习曲线、最终评估和训练好的参数
```

## 复现

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash run_10x20.sh python
```

所有运行在给定 seed 下是确定的。在 10 核的笔记本上，价值学习部分约 1 小时；交叉熵部分是串行运行的，需要十几个小时。可以用 `SEEDS` 和 `CE_WORKERS` 环境变量调整 seed 和并行进程数。

## 第三方代码

`tetris_gymnasium/` 来自 [Tetris-Gymnasium](https://github.com/Max-We/Tetris-Gymnasium)（Maximilian Weichart，MIT 许可证，见 [`LICENSE_tetris_gymnasium`](LICENSE_tetris_gymnasium)），本项目改动了其中两个文件：

- `wrappers/grouped.py`：新增 `info["game_over_mask"]` 和 `info["lines_cleared_per_action"]`，标出每个候选落点是否会输掉游戏、会消几行。
- `wrappers/observation.py`：修复用 uint8 掩码做索引、错误地把棋盘顶部两行清零的问题。

另外，原库把输掉游戏的落点编码成空棋盘，看起来像最好的结果。`t2x2/env.py` 把这些落点改为满高棋盘，这一修正没有改动库本身。
