# IncidentLens Benchmarks

This repository contains IncidentLens ground truth, experiment orchestration,
metrics, and result comparison. It does not collect data, generate simulations,
or provide the operational replay pipeline.

## Related repositories

| Repository | Responsibility |
| --- | --- |
| `urban-observations` | Collects and archives raw real-world data |
| `urban-observation-simulator` | Generates completed synthetic datasets offline from the operational pipeline |
| `urban-observation-processing` | Replays completed real or synthetic data, receives it, and performs shared enrichment |
| `incidentlens` | Detects incidents from the enriched observation stream |
| `incidentlens-benchmarks` | Owns private labels and evaluates detector results |

Ground truth remains here and is never sent to enrichment or IncidentLens.

## Install

Use Python 3.10 or newer. For full experiment orchestration, clone the related
repositories as siblings and create a benchmark-specific environment:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ../urban-observation-processing
python -m pip install -e ../incidentlens
python -m pip install -r requirements.txt
python -m pip install -e .
cp config.example.json config.json
chmod 600 config.json
```

`config.json` is ignored by Git. The example contains placeholders only.

| Setting | Meaning | Required change? |
|---|---|---|
| `paths.raw_archive_root` | Historical raw TAR root used by legacy reproduction tools. | For real-data reproduction |
| `paths.evaluation_temp_root` | Extracted temporary evaluation data. | Safe default provided |
| `paths.real_observation_cache_root` | Cached normalized real observations. | Safe default provided |
| `openai.api` | Optional historical news-labelling tool. | Only for that tool |
| `langsmith.api` | Optional tracing for historical news labelling. | No; blank disables it |
| `google_places_key.key` | Optional historical geocoding/coverage tools. | Only for those tools |

Routine scoring is host-side Python and does not require API credentials. This
repository has **no Docker containers**. The first installation uses the nine
commands above; evaluating an existing result is one Python command. A complete
synthetic experiment additionally uses one simulator command, one processing
replay command, and the running processing and IncidentLens services described
in those repositories.

## Synthetic experiments

Generate datasets separately with `urban-observation-simulator`. Once a dataset
is complete, replay it with `urban-observation-processing`:

```bash
cd ../urban-observation-processing
python -m replay.synthetic /path/to/completed/dataset
```

Run IncidentLens while that repository's receiver produces the enriched JSONL.
Then use this repository to compare predictions with the private mapping and
ground truth. Existing historical orchestration remains available through:

```bash
python -m evaluation.run_experiments --help
```

## Real-data experiments

Raw archives use one TAR per source and date:

```text
<paths.raw_archive_root>/<source>/YYYYMMDD.tar
```

The current operational replay is provided by `urban-observation-processing`.
`evaluation.real_emitter` remains here for reproducing the original benchmark
profiling and experiment procedure:

```bash
python -m evaluation.real_emitter --help
```

## Evaluation

Ground-truth files are under `evaluation/ground_truth/`. Evaluate existing
results with:

```bash
python -m evaluation.evaluate_results --help
```

The evaluator supports real and synthetic low-level incidents, compositions,
coverage, ablation, and performance experiments. Outputs are written under the
configured result/evaluation directories.

## Check the installation

```bash
python -m unittest discover -s tests -v
python -m evaluation.evaluate_results --help
```
