# Ground truth

`real/low_level_gt_corrected.json` is the curated real-data label set consumed
by low-level evaluation. Its records contain incident type, textual location,
start/end time, and the earliest article time used for reporting delay.

`real/top_level.json` contains the small high-level composition label set.

These are evaluation inputs. They must never be placed in REPORT streams,
detector prompts, caches, or IncidentLens runtime configuration.
