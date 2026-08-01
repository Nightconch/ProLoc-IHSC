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

## Fix round 1: executable validation-path coverage

- Extracted the existing validation loop without changing behavior into the private, production-specific `_run_validation_epoch` seam; main is its sole production caller.
- Replaced the accumulator-only acceptance test with a real CPU `DataLoader` harness. It processes five selected rows from a seven-row synthetic dataset at `batch_size=3`, producing a full three-row batch and an incomplete two-row batch. Using a larger backing dataset makes a processed-count divisor distinguishable from a dataset-size divisor.
- The first batch has three unique aligned proteins and loss `log(1 + 2e^-1)` at temperature 1. The second batch has one duplicated Protein ID and deliberately non-diagonal embeddings, so the canonical multi-positive loss is zero. The expected sample-weighted epoch metric remains independently derived as `3 * log(1 + 2e^-1) / 5` (`0.33086682835923065`).
- Numeric RED: with `_run_validation_epoch` temporarily changed to the old-equivalent diagonal calculation, never-incremented ISC total, and dataset-size divisor, `pytest tests\test_training_protein_aware_isc.py::test_validation_epoch_isc_is_protein_aware_and_sample_weighted -q` failed with obtained `0.0` versus expected `0.33086682835923065`.
- GREEN: after restoring the canonical accumulator call and processed-sample divisor, the same focused test passed (`1 passed`).
- Verification: protein-aware core/training plus protein-group split tests passed (`20 passed`); the full suite passed (`105 passed, 23 subtests passed`); `git diff --check` passed.
- Mutation self-review: the executable test fails if validation drops batch Protein IDs/uses diagonal targets, bypasses the accumulator, omits the final two rows, or divides by the seven-row backing dataset rather than the five processed samples.
