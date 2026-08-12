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
