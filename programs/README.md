# Programs

## `q2_decision_model.py`

Enumerates the compressed Question 2 decision set and computes expected cost and expected profit.

The program follows the handoff workflow:

- enumerate three modes for each part:
  - `first_inspect`: inspect before first assembly; do not inspect again after disassembly
  - `never_inspect`: do not inspect before assembly or after disassembly
  - `inspect_after_recovery`: do not inspect before assembly, inspect after disassembly
- combine them into `3 x 3` part policies
- compute the corresponding finished-product defect probability `q`
- decide finished-product inspection using `t_f < qL`
- enumerate disassembly `z = 0, 1`
- compare final expected profits globally

Run:

```powershell
python programs/q2_decision_model.py
```

Outputs are written to `programs/results/`.
