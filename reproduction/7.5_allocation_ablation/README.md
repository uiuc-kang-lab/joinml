# §7.5 — Adaptive-allocation ablation (Figure 10)

Goal: demonstrate that BaS's *adaptive* allocation is close to the
oracle-best fixed-α choice and well above the worst fixed-α choice. Sweeps
six datasets at one budget each, with fixed α ∈ {10%, 20%, 30%, 40%, 50%}
plus the adaptive variant.

## Run

```bash
bash reproduction/7.5_allocation_ablation/run.sh
# or:
INTERNAL_LOOP=100 MAX_CONCURRENT=5 \
    bash reproduction/7.5_allocation_ablation/run.sh
```

Submits 36 slurm jobs (6 datasets × 6 variants = 5 fixed-α + 1 adaptive),
all at `internal_loop=100`.

## Evaluate

For each dataset, compute the RRMSE-reduction vs WWJ (Figure 10's y-axis is
`(RRMSE_WWJ − RRMSE_BaS) / RRMSE_WWJ`):

```bash
# Per fixed-α + adaptive, per dataset
uv run python scripts/check_results.py runs/sec7.5-adapt/<dataset>*.jsonl
uv run python scripts/check_results.py runs/sec7.5-fixed-0.10/<dataset>*.jsonl
# ... etc for 0.20, 0.30, 0.40, 0.50
```

## Coverage

| Dataset | Budget |
|---|--:|
| Quora | 4 M |
| Company | 2 M |
| Roxford | 3 M |
| Flickr30K | 3 M |
| Webmasters | 7 M |
| VeRi | 300 k |

All six are entries in `results.py` so per-cell BaS targets are also
available for cross-comparison.
