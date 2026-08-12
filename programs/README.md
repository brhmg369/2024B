# Programs

## `q2_decision_model.py`

Enumerates the compressed Question 2 decision set and computes expected cost and expected profit.

The program follows the handoff workflow:

- enumerate four part-inspection policies `(x1, x2)`
- compute the corresponding finished-product defect probability `q`
- decide finished-product inspection using `t_f < qL`
- enumerate disassembly `z = 0, 1`
- compare final expected profits globally

Run:

```powershell
python programs/q2_decision_model.py
```

Outputs are written to `programs/results/`.
