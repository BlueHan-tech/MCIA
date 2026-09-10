# Research diagnostics archive

This directory contains exploratory audits, ablations, oracle screens, and
historical reproductions.  These scripts are not part of the main experiment
pipeline and are deliberately excluded from `test/` so that test discovery
means runnable smoke or contract checks.

- `audit_*`, `calibrate_*`, `decompose_*`, `diagnose_*`, `level0_*`,
  `preview_*`, `report_*`, and `reproduce_*` are research evidence or
  diagnostics.  They may read a specific historical run or rely on a
  superseded implementation; their filename and output metadata must be read
  before reuse.
- `retired/diagnose_db3_mcia_s05_adaptation_oracle.py` and
  `retired/diagnose_db3_task_aware_mcia.py` are frozen historical records. They refer
  to the retired target-subject pseudo-target adaptation entry and are not
  runnable under the current protocol.
- Current, reusable experiments belong in `scripts/`; runnable automated
  checks belong in `test/`.  Do not add a one-off investigation to either.

The only active in-progress attribution scripts remain in `dev/`:
`ridge_attribution.py` and `attribution_ladder.py`.
