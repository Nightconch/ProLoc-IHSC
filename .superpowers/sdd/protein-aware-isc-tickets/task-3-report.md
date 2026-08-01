# Ticket #4 report

## Files changed

- `train.py`
  - Removed the obsolete diagonal-only `isc_contrastive_loss` path.
  - Added the local `accumulate_validation_isc` seam, which computes the canonical protein-aware batch loss and updates a sample-weighted total plus actual processed count.
  - Validation now passes batch Protein IDs, uses the canonical loss, and divides ISC by the processed count (including an incomplete final batch).
- `tests/test_training_protein_aware_isc.py`
  - Added a CPU-only behavioral test using synthetic embeddings: a full batch of three unique proteins followed by an incomplete two-sample same-protein batch with non-diagonal matches. It asserts the independently hand-derived epoch ISC `3 * log(1 + 2e^-1) / 5`.

## RED / GREEN evidence

- RED: `C:\Users\87279\miniconda3\envs\psl\python.exe -m pytest tests\test_training_protein_aware_isc.py -q` produced `1 failed, 2 passed`; the focused behavior had no validation accounting seam in the prior code. The prior validation path was also verified to use the diagonal-only function and never increment `running_val_isc_loss`.
- GREEN: the same command produced `3 passed` after the canonical accumulator and validation integration.

## Verification

- `C:\Users\87279\miniconda3\envs\psl\python.exe -m pytest tests\test_protein_aware_isc.py tests\test_training_protein_aware_isc.py tests\test_train_group_split.py -q` -> `20 passed`.
- `C:\Users\87279\miniconda3\envs\psl\python.exe -m pytest -q` -> `105 passed, 23 subtests passed`.
- `git diff --check` passed.

## Self-review

- Temperature, ISC weight, classification behavior, and protein-group split logic are unchanged.
- The final validation batch contributes its actual tensor row count; the epoch divisor is the accumulated processed count.
- No production `isc_contrastive_loss` reference remains.
- The prescribed codebase graph MCP tools were unavailable in this session, so the AGENTS.md-permitted `rg` fallback was used for code discovery and removal verification.
