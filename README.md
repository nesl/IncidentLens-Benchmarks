# IncidentLens Benchmarks

This repository contains IncidentLens simulation, replay, ground truth, and
evaluation tools. The detector itself lives in `incidentlens`; collection and
shared enrichment live in `urban-observations`.

Ground truth remains here and is never sent to IncidentLens or SIGMUS.

## Repository roles

| Repository | Responsibility |
| --- | --- |
| `urban-observations` | Common observation model, receiver, and enrichment |
| `incidentlens` | Detection and optional visualization |
| `incidentlens-benchmarks` | Synthetic generation, replay, labels, and evaluation |

## Install

Clone all three repositories as siblings and use a benchmark-specific Python
3.10 or newer environment:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ../urban-observations
python -m pip install -e ../incidentlens
python -m pip install -e .
```

Create the private configuration:

```bash
cp config.example.json config.json
chmod 600 config.json
```

`config.json` is ignored by Git. Put API keys directly in it only when using
the simulator or evaluation features that require them.

## Configure replay

The `replay` section controls where data comes from and where it is sent:

```json
{
  "replay": {
    "dataset_root": "/absolute/path/to/batch_incident_runs",
    "recursive": true,
    "interval_seconds": 0.0,
    "output": null,
    "mapping_output": null,
    "receiver": {
      "enabled": true,
      "host": "127.0.0.1",
      "port": 8766,
      "timeout_seconds": 180.0
    }
  }
}
```

- `dataset_root` may be one simulator run or a directory containing runs.
- `receiver.host` and `receiver.port` identify the Urban Observations receiver.
- `interval_seconds: 0` replays at maximum speed.
- `output` optionally writes the portable observations to local JSONL.
- `mapping_output` optionally writes a private ground-truth mapping for later
  evaluation.

Command-line options override these values for one run.

## Replay synthetic data

Start the common receiver and enrichment service first:

```bash
cd ../urban-observations
./docker-start processing-up
```

Then run the configured replay:

```bash
cd ../incidentlens-benchmarks
python -m evaluation.synthetic_observations
```

The emitter converts text, time-series records, and images into the same inline
`urban-observation.v1` model used by real data. It sends one observation and
waits for one acknowledgement, providing simple backpressure. IncidentLens and
SIGMUS independently follow the enriched JSONL produced by Urban Observations.

Useful one-run overrides include:

```bash
# Replay another dataset
python -m evaluation.synthetic_observations /path/to/another/batch

# Validate conversion locally without using the receiver
python -m evaluation.synthetic_observations \
  --no-receiver --output /tmp/observations.jsonl --limit 10

# Send to a different receiver
python -m evaluation.synthetic_observations \
  --receiver-host 192.0.2.10 --receiver-port 8766
```

On the original deployment, the full synthetic archive is under:

```text
/mnt/sandia-backup/incidentlens-generated-archive/generated/batch_incident_runs/
```

## Generate synthetic data

Inspect a small generation plan without calling external services:

```bash
python -m simulator.simulator \
  --incident-types wildfire --runs-per-incident 1 \
  --output-folder smoke --max-iterations 1 --dry-run
```

Remove `--dry-run` to generate the data. Generation can use OpenAI, Google, and
geocoding APIs and may incur cost. Output is written below
`paths.simulator_output_root`.

## Real-data experiments

The raw archive is expected to contain one TAR per source and date:

```text
<paths.raw_archive_root>/<source>/YYYYMMDD.tar
```

`evaluation.real_emitter` prepares and normalizes historical data, while
`evaluation.run_experiments` coordinates historical benchmark runs. These are
reproduction tools for the older experiments; the current live path uses the
common Urban Observations receiver.

Inspect their options before a run:

```bash
python -m evaluation.real_emitter --help
python -m evaluation.run_experiments --help
```

## Evaluate results

Ground-truth files are under `evaluation/ground_truth/`. Evaluate existing
results with:

```bash
python -m evaluation.evaluate_results --help
```

The evaluator supports synthetic and real low-level incidents, compositions,
coverage, ablation, and performance experiments. Outputs are written below the
selected evaluation directory.

## Check the installation

```bash
python -m unittest discover -s tests -v
python -m evaluation.synthetic_observations --help
```

If replay reports a refused connection, confirm that Urban Observations
processing is running and that the configured host and port match its receiver.
