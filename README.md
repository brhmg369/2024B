# 2024B Question 2 Handoff Repository

This repository is organized for handing off Question 2 of the 2024 CUMCM B problem.

## Directory Layout

- `programs/`: reproducible calculation programs for Question 2.
- `docs/`: LaTeX formulas, model notes, scoring-map text, and handoff guidance for the paper writer.

Raw problem statements and grading files are kept in the workspace root for reference, but the reusable handoff material is in the two directories above.

## Quick Start

Run the Question 2 MDP solver:

```powershell
python programs/q2_decision_model.py
```

The script writes:

- `programs/results/q2_policy_results.csv`
- `programs/results/q2_state_policy.csv`
- `programs/results/q2_best_policies.csv`

The paper writer can use `docs/q2_model_handoff.md` and `docs/q2_formulas.tex` for method explanation and formula insertion.
