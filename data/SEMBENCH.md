# SemBench-derived datasets

The `ecomm-q*` and `movie-q*` datasets used by JoinML come from **SemBench**,
a benchmark of semantic query-processing engines:

> *SemBench: Benchmarking Semantic Query Processing Engines*
> https://github.com/SemBench/SemBench/

SemBench is *not* bundled inside this repository — only the pre-materialised
per-query dataset folders that JoinML reads at runtime are kept under `data/`.

## Per-query dataset layout

For each query that JoinML consumes (configured in `configs/<query>.yml`),
the expected on-disk layout is:

```
data/<query>/
├── data/table0.csv               # records for the join's first table
├── data/table1.csv               # second table (omit for self-joins)
├── data/table2.csv               # third table (multi-way joins only)
└── oracle_labels/00.csv          # ground-truth matched tuples (one tuple/line)
```

## Queries used in the JoinML paper

| Query | Source | Modality | Aggregator in paper |
|---|---|---|---|
| `ecomm-q7` | SemBench / E-commerce | text | COUNT |
| `ecomm-q8` | SemBench / E-commerce | text | MIN |
| `ecomm-q9` | SemBench / E-commerce | text | MEDIAN |
| `ecomm-q10` | SemBench / E-commerce | text (3-way join) | COUNT |
| `ecomm-q11` | SemBench / E-commerce | text (4-way join) | COUNT |
| `movie-q5` | SemBench / Movie | text | AVG |
| `movie-q6` | SemBench / Movie | text | MAX |

## Regenerating from upstream

To regenerate the per-query CSVs from SemBench instead of using the bundled
`data/<query>/` directories:

1. Clone SemBench and follow its setup instructions.
2. Run SemBench's query exporter for each `ecomm-q*` / `movie-q*` query
   you want to materialise. The exporter writes table CSVs and an oracle
   labels file in the same shape JoinML expects (see above).
3. Move or symlink the exported files into `data/<query>/`.

If you only want to *use* the queries (and not regenerate them), the bundled
CSVs under `data/<query>/` are sufficient — no SemBench install required.
