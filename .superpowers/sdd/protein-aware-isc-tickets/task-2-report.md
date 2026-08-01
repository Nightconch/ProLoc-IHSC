## Ticket #3 report

### Files changed

- `train.py`: carry normalized Protein Id values through `CustomDataset` and batches; use canonical protein-aware ISC for training; report sample-weighted actual ISC and positives per anchor. Validation only unpacks the expanded batch contract and retains its existing ISC behavior.
- `tests/test_training_protein_aware_isc.py`: synthetic dataset-to-training ISC integration coverage for indexed duplicate IDs.

### TDD evidence

- RED: `C:\Users\87279\miniconda3\envs\psl\python.exe -m pytest tests\test_training_protein_aware_isc.py -q` failed as expected because `CustomDataset` did not accept `protein_ids`.
- GREEN: the same focused test passed (`1 passed`).

### Verification

- Focused split/canonical/integration suite: `18 passed`.
- Complete local suite: `103 passed, 23 subtests passed`.
- `git diff --check` completed without whitespace errors.

### Self-review

- Temperature default (0.07), ISC weight (1.0), classification loss calculation, optimizer settings, and protein-group split behavior remain unchanged.
- Training now calls Task 1's `protein_aware_isc_loss`; the test’s swapped duplicate embeddings would be penalized by a diagonal loss.
- No validation ISC/accounting behavior was migrated; that remains Ticket #4.
- `codebase-memory-mcp` graph tools were unavailable in this session, so code discovery used the AGENTS.md-approved `rg` fallback.
