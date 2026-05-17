# Setup — datasets + proxy embeddings

This must complete before any of the §7.* experiments will run. Three steps:
(1) acquire the raw data, (2) precompute the proxy similarity tensors,
(3) smoke-test that the pipeline works end-to-end.

## 1. Acquire the raw data

The preprint uses 13 datasets across three acquisition paths.

### Auto-downloadable from the JoinML Google-Drive mirror

| Dataset | Aggregator in paper | Approx. size |
|---|---|---|
| `quora` | AVG | 5 MB |
| `company` | COUNT | 130 MB |
| `flickr30k` | COUNT | 4.2 GB |

```bash
bash reproduction/setup/download_data.sh quora company flickr30k
```

### Manual acquisition (license / external host)

| Dataset | Aggregator | Source |
|---|---|---|
| `VeRi` | AVG | https://github.com/JDAI-CV/VeRidataset (request access) |
| `roxford` | COUNT | http://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/ |
| `webmasters` | SUM | Stack Exchange data dump → Webmasters site, pre-processed per Zhang et al. 2015 |

Download from the source, then arrange under `data/<dataset>/data/table*.csv`
and `data/<dataset>/oracle_labels/00.csv`.

### SemBench-derived queries

| Dataset | Aggregator |
|---|---|
| `ecomm-q7` | COUNT |
| `ecomm-q8` | MIN |
| `ecomm-q9` | MEDIAN |
| `ecomm-q10` / `ecomm-q11` | COUNT (multi-way joins) |
| `movie-q5` | AVG |
| `movie-q6` | MAX |

These come from SemBench (https://github.com/SemBench/SemBench). See
[`../../data/SEMBENCH.md`](../../data/SEMBENCH.md) for regeneration
instructions.

### One-shot orchestration

```bash
# Acquire everything (auto-downloads what it can; prints instructions for the rest):
bash reproduction/setup/download_data.sh

# Audit which datasets are present:
bash reproduction/setup/download_data.sh check
```

### Expected layout

```
data/<dataset>/
├── data/table0.csv          # records; column "join_col" is the join key
├── data/table1.csv          # second table (omit for self-joins)
├── data/table2.csv          # third table (multi-way joins only)
└── oracle_labels/00.csv     # ground-truth match tuples "id0,id1"
```

## 2. Precompute proxy similarity tensors

The first `joinml-adapt` invocation on a `(dataset, proxy)` pair has to build
the full `|T₁| × |T₂|` similarity tensor and serialise it to disk. For the
larger datasets (Quora 28 GB, Flickr30k 38 GB) this takes hours — do it once,
ahead of time.

```bash
# Everything used in the preprint (skips datasets whose data/ folder is missing):
bash reproduction/setup/precompute_all.sh

# Or a single (dataset, proxy) pair:
uv run python reproduction/setup/precompute_embeddings.py \
    --dataset_config configs/company-minilm.yml
```

Override cache location:
```bash
CACHE_PATH=/scratch/$USER/joinml_cache bash reproduction/setup/precompute_all.sh
```

The cache is keyed by `(dataset_name, proxy)` — re-running the same
configuration is a no-op.

## 3. Verify

End-to-end smoke test (~5 min per dataset, 50 iterations):
```bash
bash reproduction/setup/verify_setup.sh
```
Reports per-dataset RRMSE; should be within ~30% of the BaS targets in
`results.py` (50 iters is too few for tight estimates — this is a sanity
check, not a measurement).

## Cache sizes (per (dataset, proxy))

Approximate disk usage for the precomputed proxy tensors:

| Dataset | Proxy | Cache size | Notes |
|---|---|---:|---|
| Company | `all-MiniLM-L6-v2` | ~750 MB | |
| Quora | `all-MiniLM-L6-v2` | ~28 GB | float64 over 3.6B pairs |
| Webmasters | `all-MiniLM-L6-v2` | ~2.4 GB | top-k pre-filtered |
| VeRi | `reid` | ~4.4 GB | |
| Roxford | `superglobal` | ~260 MB | |
| Flickr30k | `blip` | ~38 GB | |
| Ecomm-Q7 | `all-MiniLM-L6-v2` | ~240 MB | |
| Ecomm-Q8 | `clip` | ~54 MB | |
| Ecomm-Q9 | `clip` | ~148 MB | |
| Movie-Q5/Q6 | `flair` | <5 MB | |

Total (paper datasets) ≈ **75 GB**. Use a fast SSD or scratch filesystem.
