# §7.3 — Main RRMSE comparison (Figure 7)

Goal: reproduce the per-(dataset, budget) RRMSE table in `results.py` — the
"BaS" column of Figure 7 in the paper. Sixteen (dataset, aggregator)
combinations across the budget grid in `results.py`.

The linear and non-linear aggregators are split into two scripts because
non-linear aggregators (MIN/MAX/MEDIAN) need the GEV-fit allocator
(`--allocation_search evt`).

## Run

```bash
# Linear aggregators (COUNT, SUM, AVG) — Figures 7a–7h
bash reproduction/7.3_rrmse_main/linear_sweep.sh

# Non-linear aggregators (MIN, MAX, MEDIAN) — Figures 7i–7k
bash reproduction/7.3_rrmse_main/nonlinear_sweep.sh
```

Both scripts are throttled (`MAX_CONCURRENT=10`) and idempotent. Customize:
```bash
INTERNAL_LOOP=200 MAX_CONCURRENT=5 \
    bash reproduction/7.3_rrmse_main/linear_sweep.sh
```

`linear_sweep_throttled.sh` and `linear_sweep_resume.sh` are alternate entry
points for resuming a partial sweep after a maintenance window or
controller outage.

## Evaluate

Per-file RRMSE:
```bash
uv run python scripts/check_results.py runs/sec7.3/<file>.jsonl
```

Full grid comparison vs published BaS:
```bash
uv run python scripts/eval_against_targets.py --grid runs/sec7.3/*.jsonl
```

## Coverage

| Aggregator | Datasets × budgets |
|---|---|
| COUNT | Company, Roxford, Flickr30k, Ecomm-Q7/Q10/Q11 × 5 budgets each |
| SUM | Webmasters × 6 budgets |
| AVG | Quora, VeRi, Movie-Q5 × 5 budgets each |
| MIN | Ecomm-Q8 × 5 budgets |
| MAX | Movie-Q6 × 5 budgets |
| MEDIAN | Ecomm-Q9 × 5 budgets |

Total: 51 cells (41 linear + 10 non-linear; Movie-Q6 + Ecomm-Q8 + Ecomm-Q9).
