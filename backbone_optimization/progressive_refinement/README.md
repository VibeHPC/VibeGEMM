# Stage 2: progressive trajectory refinement

This component implements the paper's Stage 2 control flow while deliberately
replacing external LLM generation and GPU validation with explicit placeholders.
It does not claim that generated `.cu` files compile or are correct.

For each trajectory step the runner:

1. retrieves the selected record from `knowledge/records/`;
2. asks the replaceable provider to transform the latest validated kernel;
3. runs compilation/execution, numerical, and performance validation in order;
4. accepts candidates whose slowdown is at most 30%;
5. allows at most five repairs after a failed candidate;
6. terminates and rolls back on exhausted repairs or severe slowdown;
7. retains the best validated backbone independently across trajectories.

Run the demonstrative recovery scenario:

```bash
python backbone_optimization/progressive_refinement/refine.py
python backbone_optimization/progressive_refinement/validate.py
```

`exercise-recovery` deterministically covers repair success, repair exhaustion,
and severe-regression rollback. `all-pass` exercises the normal acceptance path.
Replace `PlaceholderLLMProvider` and `PlaceholderValidator` with real adapters
later; the orchestration and result format do not need to change.

