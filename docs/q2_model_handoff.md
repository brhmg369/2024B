# 第二问 MDP 模型交接说明

## 1. 本次版本的核心变化

第二问已经改成完整的马尔可夫决策过程（MDP）模型。原先的做法是先固定

```tex
x_1,x_2,y,z,r_1,r_2,y_r\in\{0,1\}
```

再逐个策略计算期望利润。现在的做法更灵活：把企业当前掌握的信息作为状态，把购买、检测、装配、检测成品、拆解等作为动作，由 Bellman 方程在每个状态下自动选择最优动作。

也就是说，最终策略不再是一组固定的 7 个 0-1 数，而是

```tex
\pi^\ast(b)=\arg\min_{a\in\mathcal A(b)}Q(b,a)
```

即“在状态 \(b\) 下做动作 \(a\)”的状态策略表。

## 2. 必须保留的题意修正

第二问中，“报废/拆解”的对象是不合格成品，不是零配件。零配件没有主动报废这个独立决策。

拆解后的零配件重复步骤 (1) 和步骤 (2)。步骤 (1) 对零配件只有两种处理方式：

1. 检测：若检测出不合格零配件，则丢弃；合格零配件进入装配。
2. 不检测：该零配件直接进入装配环节。

因此，不能使用截断式回收价值公式。它隐含了“检测不划算就把零配件直接丢掉”，但题目不允许零配件主动报废。只有检测出的不合格零配件才能被丢弃。

如果某零配件已经检测合格，则拆解后仍可视为合格件，因为题目说明拆解不会损坏零配件，因此重复检测是被占优的。如果某零配件首次装配前没有检测，则拆解后质量仍不确定，是否检测要纳入后续生产流程整体比较。

## 3. MDP 状态

程序采用信念状态。设当前持有零配件的真实状态为：

- `N`：没有该零配件；
- `G`：实际合格；
- `B`：实际不合格。

由于未检测零配件的真实质量不可直接观察，状态不是简单记录“好/坏”，而是记录联合概率分布：

```tex
b(\omega_1,\omega_2)
=
P(\Omega_1=\omega_1,\Omega_2=\omega_2\mid \text{已有信息}).
```

这里用联合分布而不是两个独立概率，是因为拆解后已知“成品不合格”，这会使两个零配件质量产生相关性。只存两个边际概率会漏掉这种关联。

## 4. MDP 动作

在每个状态下，程序自动生成可行动作：

- `buy_p1_test` / `buy_p2_test`：购买并检测零配件，坏件丢弃，直到获得合格件。
- `buy_p1_notest` / `buy_p2_notest`：购买零配件但不检测，直接持有质量不确定的零配件。
- `inspect_p1` / `inspect_p2`：检测已经持有但质量不确定的零配件；检测出坏件后丢弃。
- `assemble_notest_scrap`：装配，不检测成品，不合格后报废。
- `assemble_notest_disassemble`：装配，不检测成品，不合格后拆解。
- `assemble_test_scrap`：装配，检测成品，不合格后报废。
- `assemble_test_disassemble`：装配，检测成品，不合格后拆解。

原来的 \(x_1,x_2,y,z,r_1,r_2,y_r\) 可以作为这些动作的解释标签：

- 首次缺件时选择 `buy_pi_test` 或 `buy_pi_notest`，对应 \(x_i=1\) 或 \(x_i=0\)。
- 拆解回收后对不确定零配件选择 `inspect_pi` 或直接装配，对应 \(r_i=1\) 或 \(r_i=0\)。
- 装配动作中的 `test/notest` 对应 \(y\) 或 \(y_r\)。
- 装配动作中的 `scrap/disassemble` 对应 \(z=0\) 或 \(z=1\)。

## 5. Bellman 方程

令 \(V(b)\) 表示从状态 \(b\) 出发，为交付 1 件合格成品所需的最小期望成本。则：

```tex
V(b)
=
\min_{a\in\mathcal A(b)}
\left[
C(b,a)+
\sum_{b'}P(b'\mid b,a)V(b')
\right].
```

状态--动作值函数为：

```tex
Q(b,a)
=
C(b,a)+
\sum_{b'}P(b'\mid b,a)V(b').
```

程序比较同一状态下所有可行动作的 \(Q(b,a)\)，取最小者作为最优动作。

最终期望利润为：

```tex
\Pi=S-V(b_0),
```

其中 \(b_0\) 是初始状态，即两个零配件都没有。

## 6. 局部判断仍然可以写，但不是最终依据

如果只看某一轮成品检测，仍可使用：

```tex
t_f<qL
```

其中 \(q\) 是该轮成品不合格概率。这个式子表示：检测成品的成本是否小于它能避免的期望调换损失。

但这只是局部判断。对于拆解后首次未检测的零配件，质量不确定会影响后续多轮生产，因此不能只用局部回收价值或一次检测收益判断，必须进入 MDP 的总期望成本递推。

一句话保留：

```tex
\boxed{\text{局部判断可以减少理解难度，但最终最优策略仍要靠总期望利润比较。}}
```

## 7. 程序与输出文件

程序文件：

```text
programs/q2_decision_model.py
```

运行：

```powershell
python programs/q2_decision_model.py
```

输出：

```text
programs/results/q2_best_policies.csv
programs/results/q2_state_policy.csv
programs/results/q2_policy_results.csv
```

`q2_best_policies.csv`：6 种情况的最优期望成本、期望利润，以及几个关键状态的最优动作。

`q2_state_policy.csv`：所有可达状态下的最优动作，即 MDP 的状态策略表。

`q2_policy_results.csv`：所有可达状态下所有可行动作的 \(Q\) 值，用于画图或解释“为什么这个状态选这个动作”。

程序中对信念状态做了数值离散化：概率保留到 \(10^{-6}\)，Bellman 迭代收敛阈值为 \(10^{-10}\)。若后续论文手想画策略图，优先使用 `q2_state_policy.csv`。

注意：同一状态可能出现多个动作的 \(Q\) 值完全相同，例如先买零配件 1 还是先买零配件 2 只是顺序不同。`q2_policy_results.csv` 中 `is_optimal=1` 表示该动作并列最优，`chosen_by_tiebreak=1` 表示程序在并列时选作代表的动作。

在 MDP 中，程序不是强迫某个动作无限重复，而是在每个状态重新选择最优动作。因此旧版“固定策略不可行”的列已不再适用。若某个非最优动作在当前状态下反复执行会造成无穷循环，程序用 `self_loop_if_repeated=1` 标记它，但 Bellman 方程通常会避开这类动作。

## 8. 当前 6 种情况的最优结果

运行程序后得到的摘要如下：

| 情况 | 期望成本 | 期望利润 | 初始动作 | 两个新零件均未检测时 | 两个零件已知合格时 |
|---:|---:|---:|---|---|---|
| 1 | 37.077779 | 18.922221 | buy_p1_test | inspect_p1 | assemble_notest_disassemble |
| 2 | 44.000000 | 12.000000 | buy_p1_test | inspect_p1 | assemble_notest_disassemble |
| 3 | 39.346664 | 16.653336 | buy_p1_notest | assemble_test_disassemble | assemble_notest_disassemble |
| 4 | 41.250000 | 14.750000 | buy_p1_test | inspect_p1 | assemble_test_disassemble |
| 5 | 40.550000 | 15.450000 | buy_p2_test | inspect_p2 | assemble_notest_disassemble |
| 6 | 34.321330 | 21.678670 | buy_p1_notest | assemble_notest_scrap | assemble_notest_scrap |

注意：初始动作只是第一步动作，不等于完整策略。完整策略要看 `q2_state_policy.csv`。

## 9. 论文写作建议

论文手可以按下面顺序写第二问：

1. 先说明题意修正：零配件不能主动报废，只有检测出的坏件才能丢弃。
2. 解释为什么从固定枚举升级为 MDP：拆解后状态会改变，最优动作应随状态变化。
3. 定义信念状态 \(b(\omega_1,\omega_2)\)。
4. 定义购买、检测、装配、成品检测、拆解动作。
5. 写出不合格成品拆解后的贝叶斯更新公式。
6. 写出 Bellman 方程和 \(Q(b,a)\)。
7. 用 `q2_best_policies.csv` 展示 6 种情况结果。
8. 使用 `code/problem2_figures.py` 已生成的逐问 Figure Pack：策略路径、值迭代/离散精度验证和三类参数敏感性图。

当前正式图及源数据位于：

- `figures/q2/fig_q2_policy_path.pdf` / `.svg`；
- `figures/q2/fig_q2_validation.pdf` / `.svg`；
- `figures/q2/fig_q2_sensitivity.pdf` / `.svg`；
- `figures/q2/data/*.csv`。

图题、引入句、解释和 Figure QA 状态统一记录在 `reports/FIGURE_MANIFEST.md`。
