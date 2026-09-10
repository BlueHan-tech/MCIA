# Retired research code

These files are retained only for historical traceability. They are excluded
from `scripts/`, are not supported entry points, and must not create new
results.

- `02_finetune_mcia_db3_amputee.py`,
  `03_generate_augmented_db3_semg.py`, `generate_paper_figures.py`, and
  `generate_tables_from_run.py` depend on invalid target-subject pseudo-target
  adaptation or its derived outputs.
- `diagnose_db3_mcia_s05_adaptation_oracle.py` and
  `diagnose_db3_task_aware_mcia.py` are frozen diagnostics built on that same
  retired path and are not runnable under the current protocol.
- `paper_figures.py` is the corresponding obsolete rendering/table helper.

Historical run outputs remain evidence of their own run only; none of these
files belongs to the current main experiment or baseline comparison.
