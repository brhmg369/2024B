# Programs

## `q2_decision_model.py`

Solves Question 2 with a belief-state Markov decision process.

The program tracks the joint probability distribution of the two held parts'
qualities, then uses Bellman value iteration to choose the best action in each
state.

Run:

```powershell
python programs/q2_decision_model.py
```

Outputs are written to `programs/results/`:

- `q2_best_policies.csv`: best cost/profit summary for the six cases.
- `q2_state_policy.csv`: optimal MDP action for every reachable state.
- `q2_policy_results.csv`: all state-action Q values.

## `q3_decision_model.py`

Solves Question 3 with an analytical Markov reward process for fixed 16-bit
strategies, then validates a genetic algorithm with full enumeration.

Run:

```powershell
python programs/q3_decision_model.py
```

Outputs are written to `programs/results/`:

- `q3_summary.txt`: GA, exact enumeration, and Monte Carlo check summary.
- `q3_ga_runs.csv`: 20 independent GA runs.
- `q3_top10_strategies.csv`: top 10 exact-enumeration strategies.
- `q3_exact_all_strategies.csv`: all 65536 exact strategy evaluations.
