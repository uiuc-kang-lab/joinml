# §7.2 — Statistical guarantees (Figure 5)

Goal: verify that `joinml-adapt`'s bootstrap-t confidence intervals achieve
the nominal 95% coverage at small and large oracle budgets, across
COUNT/SUM/AVG/MEDIAN aggregators.

## Run

```bash
bash reproduction/7.2_statistical_guarantees/run.sh
# or, customising:
INTERNAL_LOOP=500 MAX_CONCURRENT=5 \
    bash reproduction/7.2_statistical_guarantees/run.sh
```

Submits 14 slurm jobs (one per Figure-5 cell) at `internal_loop=500` with
bootstrap-t CIs, throttled to 10 concurrent. Each job runs on the
`scripts/slurm_bas.sh` wrapper.

## Evaluate

```bash
for f in runs/sec7.2/*.jsonl; do
    echo "=== $(basename "$f") ==="
    uv run python scripts/check_guarantees.py "$f"
done
```

**Acceptance:** the 95-th-percentile of `|est − gt| / (CI half-width)` across
the 500 runs must be ≤ 1 for every (dataset, budget) cell. This is the
definition of a valid 95% CI: at most 5% of runs may have `|est − gt|`
exceeding the CI half-width.

## Outputs

- `runs/sec7.2/*.jsonl` — one file per (dataset, aggregator) configuration;
  each line is one independent run.
- Logs under `logs/` (one per slurm job).
