# IncidentLens Benchmarks

This repository contains the reproducibility infrastructure for IncidentLens:
synthetic incident generation, real-data replay, baseline orchestration,
ground-truth labels, and result evaluation.

The associated IncidentLens paper is included at
[`docs/incidentlens.pdf`](docs/incidentlens.pdf). The detector itself lives in
the separate `incidentlens` repository.

## How the repositories work together

- `urban-observations` provides the normalized `REPORT` schema.
- `incidentlens` receives report streams and writes predictions.
- `incidentlens-benchmarks` owns simulator plans, emitters, labels, experiment
  scheduling, and metrics.

Ground truth is never sent to the detector. Synthetic plans and real labels
remain in this repository while emitters send only normalized observations and
sensor metadata.

## Requirements and installation

Use Python 3.10 or newer. Clone `urban-observations`, `incidentlens`, and this
repository as sibling directories, then create one shared environment from
their parent directory:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

python -m pip install --no-deps -e ./urban-observations
python -m pip install -r ./incidentlens/requirements.txt
python -m pip install --no-deps -e ./incidentlens
python -m pip install -r ./incidentlens-benchmarks/requirements.txt
python -m pip install --no-deps -e ./incidentlens-benchmarks
```

The simulator can make paid OpenAI and Google model/API calls. Evaluation of
existing result files is local unless geographic matching is configured to
geocode uncached textual locations.

## Configure paths and credentials

From `incidentlens-benchmarks/`:

```bash
cp config.example.json config.json
cp .env.example .env
chmod 600 config.json .env
```

Edit `.env` and use absolute paths:

```dotenv
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=
GOOGLE_API_KEY=your_google_genai_key
GOOGLE_PLACES_API_KEY=your_google_maps_geocoding_key
NEO4J_PASSWORD=your_neo4j_password
INCIDENTLENS_RAW_ARCHIVE_ROOT=/mnt/urban-backup/raw
INCIDENTLENS_PULLED_DATA_ROOT=/mnt/urban-data/pulled_data
INCIDENTLENS_EVALUATION_TEMP_ROOT=/mnt/incidentlens/evaluation-temp
INCIDENTLENS_SIMULATOR_OUTPUT_ROOT=/mnt/incidentlens/simulator-output
INCIDENTLENS_RUNTIME_ROOT=/mnt/incidentlens/runtime
INCIDENTLENS_RESULTS_ROOT=/mnt/incidentlens/results
INCIDENTLENS_OBSERVATION_CACHE_ROOT=/mnt/incidentlens/cache/by_incident
INCIDENTLENS_REAL_OBSERVATION_CACHE_ROOT=/mnt/incidentlens/cache/real_data
```

`OPENAI_API_KEY` drives the simulator and IncidentLens model.
`GOOGLE_API_KEY` is used for cloud image editing, while
`GOOGLE_PLACES_API_KEY` is used for geocoding. `LANGSMITH_API_KEY` can remain
empty. Neo4j is used by simulator tool workflows; it is not needed to calculate
metrics from existing results.

Load configuration in every terminal:

```bash
set -a
. ./.env
set +a
export URBAN_SYSTEM_CONFIG="$PWD/config.json"
```

The same environment is inherited when `evaluation.run_experiments` launches
IncidentLens as a child process. `.env` and `config.json` are ignored by Git.

## Test both projects together

Run the repository tests:

```bash
python -m unittest discover -s tests -v
```

Then verify that the orchestrator can construct a paired detector/emitter run
without starting services or calling APIs:

```bash
python -m evaluation.run_experiments \
  --exp-type synth \
  --only-baseline direct_observation \
  --no-include-composition \
  --dry-run \
  --no-progress-bar
```

The dry run prints the exact `detection.full_pipeline` and
`evaluation.synthetic_emitter` commands it would launch.

## Generate synthetic data

First inspect a small plan without loading configuration or calling services:

```bash
python -m simulator.simulator \
  --incident-types wildfire \
  --runs-per-incident 1 \
  --output-folder smoke \
  --max-iterations 1 \
  --dry-run
```

Remove `--dry-run` to generate it:

```bash
python -m simulator.simulator \
  --incident-types wildfire \
  --runs-per-incident 1 \
  --output-folder smoke \
  --max-iterations 1
```

For a minimal multimodal smoke run containing CCTV image, weather time-series,
and news text observations, use an isolated output root:

```bash
SIMULATOR_INSIDE_SENSORS_PER_REGION=1 \
SIMULATOR_OUTSIDE_SENSORS_ENABLED=false \
SIMULATOR_MAX_IMAGE_EDITS_PER_STEP=1 \
python -m simulator.simulator \
  --incident-types wildfire \
  --runs-per-incident 1 \
  --output-root /tmp/incidentlens-synthetic-smoke \
  --output-folder one-each \
  --max-iterations 1 \
  --sources cctv weather news
```

`--sources` is intended for reproducible smoke tests. Without it, the planning
model selects sources appropriate to each simulated incident step.
The image editor defaults to the generally available `gemini-2.5-flash-image`;
set `SIMULATOR_IMAGE_EDIT_MODEL` if your Google Cloud project uses another
compatible image-editing model.

Output is written beneath `${INCIDENTLENS_SIMULATOR_OUTPUT_ROOT}`:

```text
smoke/
├── batch_run_schedule.json
└── wildfire1/
    ├── observations.txt
    ├── *_gt_*.json
    ├── *_plan.json
    └── generated image, time-series, and text files
```

Simulation uses external models and may incur cost. Image generation/editing
can be disabled for a run with:

```bash
SIMULATOR_SIMULATE_IMAGES=false python -m simulator.simulator \
  --incident-types wildfire --runs-per-incident 1 \
  --output-folder smoke-no-images --max-iterations 1
```

Validate generated ground-truth geometry:

```bash
python -m simulator.tools.validate_data \
  --root "$INCIDENTLENS_SIMULATOR_OUTPUT_ROOT/smoke" \
  --fail-on-invalid
```

## Replay synthetic data without IncidentLens

This converts simulator observations into versioned normalized reports without
opening a socket:

```bash
python -m evaluation.synthetic_emitter \
  --batch-root "$INCIDENTLENS_SIMULATOR_OUTPUT_ROOT/smoke" \
  --recursive-discovery \
  --no-emit-to-socket \
  --write-reports-to-incident-folders \
  --no-display-ground-truth \
  --no-display-sensor-locations
```

## Run a paired synthetic experiment

`run_experiments` starts IncidentLens, waits for its TCP socket, runs the
emitter, captures logs, and stops the detector when the stream completes.
For a generated batch:

```bash
python -m evaluation.run_experiments \
  --exp-type synth \
  --only-baseline incidentlens \
  --no-include-composition \
  --low-level-batch-root "$INCIDENTLENS_SIMULATOR_OUTPUT_ROOT/smoke" \
  --low-level-incident-types evaluation/incident_list_synth_batch.txt \
  --low-level-results-folder smoke
```

Use `--only-baseline direct_observation` for a simpler baseline run. Child logs
are stored under `evaluation/experiment_logs/`. Predictions are written under
`${INCIDENTLENS_RESULTS_ROOT}/<method>/<experiment>/`.

## Real-data archive and replay

The immutable archive uses one TAR per source and date:

```text
${INCIDENTLENS_RAW_ARCHIVE_ROOT}/
├── air_data/YYYYMMDD.tar
├── alertcalifornia/YYYYMMDD.tar
├── cctv/YYYYMMDD.tar
├── citizen_data/YYYYMMDD.tar
├── gkg/YYYYMMDD.tar
├── pem_data_chp_incidents_day/YYYYMMDD.tar
├── pem_data_station_5min/YYYYMMDD.tar
├── twitter_data/YYYYMMDD.tar
└── weather_data/YYYYMMDD.tar
```

`evaluation.real_emitter` can extract requested TARs into:

```text
${INCIDENTLENS_EVALUATION_TEMP_ROOT}/<source>/YYYYMMDD/...
```

Profile and normalize one date without sending it to IncidentLens:

```bash
python -m evaluation.real_emitter \
  --dates 20250110 \
  --raw-root "$INCIDENTLENS_RAW_ARCHIVE_ROOT" \
  --temp-root "$INCIDENTLENS_EVALUATION_TEMP_ROOT" \
  --auto-prepare-temp-data \
  --no-emit-to-socket \
  --write-ordered-reports-jsonl
```

For automated real experiments, inspect all options with:

```bash
python -m evaluation.run_experiments --help
```

## Ground truth

The authoritative real low-level labels are:

```text
evaluation/ground_truth/real/low_level_gt_corrected.json
```

Each entry records an incident ID and name, canonical type, textual location,
active start/end datetimes, and earliest external article datetime. Real
composition labels are in:

```text
evaluation/ground_truth/real/top_level.json
```

`evaluation/filtered_incidents.txt` is the allowed detector vocabulary, not a
label file. Keeping labels in this repository prevents the detector from
accessing ground truth during replay.

## Evaluate results

Evaluate synthetic low-level predictions:

```bash
python -m evaluation.evaluate_results \
  --mode low_incident \
  --results-root "$INCIDENTLENS_RESULTS_ROOT" \
  --gt-roots "$INCIDENTLENS_SIMULATOR_OUTPUT_ROOT/smoke" \
  --incident-types evaluation/incident_list_synth_batch.txt \
  --output-dir evaluation/evaluation_summary
```

Evaluate real-data predictions against the tracked labels:

```bash
python -m evaluation.evaluate_results \
  --mode low_real \
  --results-root "$INCIDENTLENS_RESULTS_ROOT" \
  --real-gt-path evaluation/ground_truth/real/low_level_gt_corrected.json \
  --real-no-spatial-matching \
  --no-real-geocode-gt-locations \
  --output-dir evaluation/evaluation_summary
```

Those two flags provide an offline type-and-time evaluation. Omit them when you
want spatial matching and localization metrics; that mode uses the configured
geocoding services or an existing geographic cache.

The evaluator writes JSON and CSV summaries beneath the selected output
directory. Use `python -m evaluation.evaluate_results --help` for composition,
coverage, ablation, throughput, and scalability modes.

## Troubleshooting

- `No module named detection`: install the sibling IncidentLens checkout in the
  active environment.
- `No module named observation_contract`: install the sibling Urban
  Observations checkout.
- Missing configuration variable: load `.env` with `set -a` before sourcing it.
- Socket connection refused: start IncidentLens first or use
  `evaluation.run_experiments`, which manages process order.
- No synthetic runs discovered: pass the directory containing incident folders
  and add `--recursive-discovery` when runs are nested more deeply.
- No real observations found: verify the exact
  `<raw-root>/<source>/YYYYMMDD.tar` layout.
