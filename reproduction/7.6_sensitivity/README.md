# §7.6 — Sensitivity analysis (Figures 13a, 13b)

Goal: characterise BaS's robustness to its two main hyperparameters — the
maximum blocking ratio α and the weight exponent γ.

| Script | Figure | What varies |
|---|---|---|
| `alpha_ratio.sh` | 13a | `--max_blocking_ratio α` ∈ {0.10, 0.15, 0.20, 0.25, 0.30} on Webmasters (paper) and Flickr30k |
| `weight_exponent.sh` | 13b | `--w_exp γ` ∈ {0.5, 1, 1.5, 2, 3, 5} on Company |
| `min_max_proposals.sh` | (extra) | MIN/MAX allocator proposals (uniform, EVT-fit, force-block, larger-α) on Ecomm-Q8 — this is the experiment that produced RRMSE 0.00 in our reproduction |
| `ablation_focused.sh` | (extra) | Outlier-cell knob ablations: `defmix=0.01`, `allocation_search=subset`, both, `strata_size=500` |
| `ablation_full.sh` | (extra) | Broader sweep over the same knobs across more datasets |

## Run

```bash
# Paper figures
bash reproduction/7.6_sensitivity/alpha_ratio.sh
bash reproduction/7.6_sensitivity/weight_exponent.sh
```

Each script throttles to 10 concurrent slurm jobs by default
(`MAX_CONCURRENT=N` to override).

## Evaluate

For each α / γ sweep, compute RRMSE per variant:
```bash
for d in runs/sec7.6-alpha-*/; do
    echo "=== $d ==="
    uv run python scripts/check_results.py "$d"*.jsonl
done

for d in runs/sec7.6-wexp-*/; do
    echo "=== $d ==="
    uv run python scripts/check_results.py "$d"*.jsonl
done
```
