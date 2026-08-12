# 2024 年国赛 B 题建模仓库

仓库现包含问题一抽样检测程序、问题二决策模型、问题三遗传算法与全枚举模型，以及可持续编译的 XeLaTeX 论文环境。

## Directory Layout

- `code/`: 问题一的精确二项/超几何抽样程序与测试。
- `programs/`: 问题二联合信念状态 MDP、问题三 GA/枚举模型与原始状态/动作结果。
- `reports/`: 建模、结果、Figure Manifest、逐问写作状态和论文验收报告。
- `results/`: 问题一正式结果及问题二基准比较结果。
- `figures/q1/`、`figures/q2/`: 按问题组织的源数据、PDF/SVG Figure Pack。
- `docs/`: 问题二、问题三模型说明和兼容交接入口。
- `paper/`: XeLaTeX 论文源码。

## 问题一：抽样检测

默认参数为标称次品率 10%、拒收信度 95%、接收信度 90%。运行二项主模型：

```powershell
python -m pip install -r requirements.txt
python code/problem1.py --max-sample-size 200
```

题面没有给出有限总体量 `N`，所以默认正式结果采用二项分布。实际 `N` 已知且为无放回抽样时，可同时计算超几何有限总体修正：

```powershell
python code/problem1.py --population-size 1000 --sample-size 22 --observed-defects 0
```

其中 `1000`、`22`、`0` 分别替换为实际的 `N`、`n`、`x`。程序输出：

- `results/q1_thresholds.csv`
- `results/q1_sensitivity.csv`
- `results/q1_population_sensitivity.csv`
- `results/q1_summary.json`
- `figures/q1/*.pdf`
- `figures/q1/*.svg`
- `figures/q1/data/*.csv`

运行测试：

```powershell
python -m unittest discover -s code -p "test_*.py" -v
```

建模口径和结果解释见 `reports/ANALYSIS_MODELING_REPORT.md`、`reports/RESULTS_REPORT.md`。

## 问题二：生产决策

运行问题二联合信念状态 MDP 求解器与 Figure Pack：

```powershell
python programs/q2_decision_model.py
python code/problem2_figures.py
```

程序输出：

- `programs/results/q2_policy_results.csv`
- `programs/results/q2_state_policy.csv`
- `programs/results/q2_best_policies.csv`
- `programs/results/q2_convergence.csv`
- `programs/results/q2_summary.json`
- `figures/q2/fig_q2_*.pdf`
- `figures/q2/fig_q2_*.svg`
- `figures/q2/data/*.csv`

问题二模型说明见 `docs/q2_model_handoff.md`；论文正式章节为 `paper/sections/6_problem2.tex`。

## 问题三：生产流程优化

运行问题三遗传算法与全枚举模型：

```powershell
python programs/q3_decision_model.py
python code/problem3_figures.py
python code/problem3_schematic.py
python code/problem3_sensitivity.py
python code/problem3_mc_check.py
```

程序输出：

- `programs/results/q3_summary.txt`
- `programs/results/q3_ga_runs.csv`
- `programs/results/q3_top10_strategies.csv`
- `programs/results/q3_exact_all_strategies.csv`
- `programs/results/q3_mc_stats.json`
- `figures/q3/`：6 幅正式图（结构、失效循环、利润分布、前 10、GA 收敛、敏感性）与源数据

问题三模型说明见 `docs/q3_model_handoff.md`，公式见 `docs/q3_formulas.tex`。

## 问题四：抽样不确定性下的重决策

运行问题四 Beta 后验情景重决策（七变量固定策略类 + 16 位策略类）与收敛图：

```powershell
python programs/q4_bayesian_model.py --scenarios 10000
python code/problem4_figures.py
```

先验敏感性与情景数稳定性实验（输出到临时目录，不覆盖正式结果）：

```powershell
python programs/q4_bayesian_model.py --scenarios 10000 --sample-sizes 40 --prior jeffreys --output-dir tmp/q4_experiments/jeffreys_n40
python programs/q4_bayesian_model.py --scenarios 1000 --sample-sizes 40 --output-dir tmp/q4_experiments/S1000
python programs/q4_bayesian_model.py --scenarios 5000 --sample-sizes 40 --output-dir tmp/q4_experiments/S5000
python programs/q4_bayesian_model.py --scenarios 20000 --sample-sizes 40 --output-dir tmp/q4_experiments/S20000
```

问题四结果文件位于 `programs/results/q4_*.csv` 与 `q4_summary.json`；抽样样本量 `n` 为证据强度参数，`n=40` 仅为代表性小样本情景。

## LaTeX 论文环境

`paper/` 下为 XeLaTeX 论文源码，版式遵循 2024 年国赛格式参考。

在仓库根目录编译：

```powershell
.\build.ps1
```

PDF 输出到 `output/pdf/main.pdf`。需要监控源码改动并自动重编译时运行：

```powershell
.\watch.ps1
```

VS Code 中安装推荐的 LaTeX Workshop 扩展后，保存 `.tex` 会自动触发同一构建；`Ctrl+Shift+B` 也可运行仓库构建任务。使用 `.\build.ps1 -Clean` 清理中间文件。

常用编辑入口：

- `paper/main.tex`：标题、关键词和章节装配入口。
- `paper/sections/abstract.tex`：随各问题进度增量维护的摘要。
- `paper/sections/1_restatement.tex`：问题重述。
- `paper/sections/2_analysis.tex`：问题分析与各问关系。
- `paper/sections/3_assumptions.tex`：模型假设及适用范围。
- `paper/sections/4_symbols.tex`：符号、类型和单位。
- `paper/sections/5_problem1.tex`：模型建立与求解入口；5.1 问题一完整闭环（含敏感性）。
- `paper/sections/6_problem2.tex`：5.2 问题二联合信念状态、Bellman 方程、结果与验证。
- `paper/sections/7_problem3.tex`：5.3 问题三多工序期望递推、GA、结果与验证。
- `paper/sections/8_problem4.tex`：5.4 问题四 Beta 后验情景重决策、结果与验证。
- `paper/sections/9_evaluation.tex`：模型评价、改进与推广。
- `paper/sections/A_code.tex`：复现说明和程序附录。

逐问完成状态、结果来源、假设、图表、风险及运行版本统一记录在 `reports/PAPER_WRITING_STATE.md`。四问均已形成模型、结果与论文章节；由于实际抽样样本量仍需企业确认，当前文件属于工作稿，不应作为全题终稿提交。

`format2024.doc` 中的承诺书和编号专用页属于提交表单，不放入电子论文 PDF。
