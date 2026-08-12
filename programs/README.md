# Programs

## `q2_decision_model.py`

Enumerates the rigorous Question 2 decision set and computes expected cost and expected profit.

The program follows the handoff workflow:

- enumerate seven binary decisions `(x1, x2, y, z, r1, r2, yr)`
- track recovered parts as `known_good`, `unknown_good`, or `unknown_bad`
- solve a linear expectation system for each policy
- mark singular or infinite-loop policies as infeasible
- compare feasible non-dominated policies by expected profit

Run:

```powershell
python programs/q2_decision_model.py
```

Outputs are written to `programs/results/`.
