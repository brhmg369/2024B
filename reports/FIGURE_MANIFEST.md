# Figure Manifest

所有正式图均使用 `figures/style/cumcm.mplstyle`，数据型线图同时导出 PDF/SVG；Figure ID 与最终论文图号解耦。

| Figure ID | 问题 | 论证标签 | 该图说明 | 源数据 | 绘图代码/源文件 | 输出文件 | Caption | Lead-in | Interpretation | 最终宽度 | QA | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1-BOUNDARY | Q1 | [RESULT] | 二项模型下接收上界始终低于拒收下界，`n=22` 首次形成统一可执行方案。 | `figures/q1/data/q1_thresholds.csv` | `code/problem1.py` | `figures/q1/q1_decision_boundaries.pdf`, `.svg` | 二项模型的接收与拒收临界值随样本量的变化 | 为检验三态判定边界，绘制临界值随样本量的变化。 | 接收域与拒收域不重叠；`n=22` 时分别为 0 与 6。 | `0.90\textwidth` | 数据、矢量、中文、缩小、灰度、引入和解释均通过 | ready |
| Q1-CONFIDENCE | Q1 | [SENS] | 接收方向的最小样本量随信度提高显著增加，而极端拒收指标仅作辅助解释。 | `figures/q1/data/q1_sensitivity.csv` | `code/problem1.py` | `figures/q1/q1_confidence_sensitivity.pdf`, `.svg` | 置信水平变化对方向性最小样本量的影响 | 为量化置信要求变化的影响，比较两类方向性样本量。 | 接收信度从 0.80 增至 0.99 时，样本量从 16 增至 44。 | `0.96\textwidth` | 数据、矢量、中文、缩小、灰度、引入和解释均通过 | ready |
| Q1-POPULATION | Q1 | [VALIDATE] | 总体量增大时，超几何统一样本量收敛到二项极限 22。 | `figures/q1/data/q1_population_sensitivity.csv` | `code/problem1.py` | `figures/q1/q1_population_sensitivity.pdf`, `.svg` | 有限总体量变化下超几何模型与二项极限的比较 | 为检查二项近似的适用范围，比较不同总体量的超几何结果。 | `N=1000` 与 `5000` 时均得到 22，与二项结果一致。 | `0.88\textwidth` | 数据、矢量、对数轴标注、灰度、引入和解释均通过 | ready |
| Q2-POLICY | Q2 | [RESULT] | 六种情形在首次购件、装配及首次缺陷后的状态相关动作不同。 | `figures/q2/data/q2_policy_path.csv` | `code/problem2_figures.py` | `figures/q2/fig_q2_policy_path.pdf`, `.svg` | 六种参数情形从初始状态到首次缺陷后的最优动作路径 | 为把状态策略翻译为可执行动作，比较六种情形的关键路径。 | 情形 3 利用成品检测后的信息反馈再检部件 1；情形 6 直接报废并重购。 | `0.98\textwidth` | 数据、矢量、颜色语义、文字冗余编码、缩小和论文一致性通过 | ready |
| Q2-VALIDATION | Q2 | [VALIDATE] | 值迭代快速收敛，六位信念离散相对七位结果的成本误差远低于正文精度。 | `figures/q2/data/q2_convergence.csv`, `q2_rounding_validation.csv` | `code/problem2_figures.py` | `figures/q2/fig_q2_validation.pdf`, `.svg` | 问题二值迭代收敛与信念状态离散精度检验 | 为验证求解器停止条件和状态合并精度，绘制迭代与精度结果。 | 23--29 轮收敛；最大 Bellman 残差 `2.594e-11`，六位成本差 `3.227e-6` 元。 | `0.98\textwidth` | 数据、矢量、对数轴、线型/marker 冗余和缩小测试通过 | ready |
| Q2-SENSITIVITY | Q2 | [SENS] | 次品率对利润及策略切换影响最明显；损失或拆解成本的影响取决于相应动作是否启用。 | `figures/q2/data/q2_sensitivity.csv` | `code/problem2_figures.py` | `figures/q2/fig_q2_sensitivity.pdf`, `.svg` | 次品率、调换损失与拆解成本扰动下的期望利润 | 为检验基准策略在参数邻域内的稳定性，进行三类单因素扰动。 | 次品率使四种情形切换；调换损失只使情形 5 切换，拆解成本只使情形 3 切换。 | `0.99\textwidth` | 数据、统一坐标、线型/marker 冗余、灰度、缩小和解释通过 | ready |
