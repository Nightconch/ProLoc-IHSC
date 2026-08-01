# Task 1 report — Issue #2 protein-aware multi-positive ISC

## What changed

- Added the small canonical seam in `isc.py`:
  - `protein_positive_relation(protein_ids)` trims IDs, compares them case-sensitively, rejects missing or blank values, and returns the full boolean same-protein relation.
  - `protein_aware_isc_loss(image_embeddings, sequence_embeddings, protein_ids, temperature=0.07)` normalizes embeddings, aggregates the log-probability mass of every same-protein positive in both directions, and averages the directional losses.
  - Explicit validation rejects mismatched embedding or identifier counts; the multi-positive reduction also guards against a row with no positives.
- Added focused CPU-only behavioral coverage. The test suite verifies whitespace trimming and case sensitivity, invalid IDs, count mismatch, all-same zero loss, unique-ID equivalence to the existing diagonal ISC objective, finite mixed-ID gradients, and same-protein permutation invariance.
- Did not modify training or validation loops.

## Files changed

- `isc.py`
- `tests/test_protein_aware_isc.py`

## TDD evidence

### RED

Command:

```powershell
rtk proxy C:\Users\87279\miniconda3\envs\psl\python.exe -m pytest tests\test_protein_aware_isc.py -q
```

Result: collection failed as expected with `ModuleNotFoundError: No module named 'isc'` (7 focused behavior tests had been written before the implementation existed). This proves the test required the missing canonical seam.

### GREEN

Command:

```powershell
rtk proxy C:\Users\87279\miniconda3\envs\psl\python.exe -m pytest tests\test_protein_aware_isc.py -q
```

Result: `7 passed in 2.70s`.

## Verification

Command:

```powershell
rtk proxy C:\Users\87279\miniconda3\envs\psl\python.exe -m pytest -q
```

Result: `100 passed, 23 subtests passed in 12.25s`; repeated immediately before commit with `100 passed, 23 subtests passed in 12.28s`.

Also ran `rtk git diff --check`; it completed without output or errors.

## Self-review

- The reduction is `logsumexp(all pairs) - logsumexp(all positive pairs)`, so it uses all same-ID positive mass rather than selecting a diagonal pair.
- With all IDs the same, numerator and denominator match in both directions, producing zero loss.
- With unique IDs, each positive set is exactly one diagonal entry, reproducing the existing symmetric cross-entropy objective.
- The public implementation is function-only and has no registry, configuration, compatibility layer, or training-flow changes.

## Fix round 1 — numeric Protein IDs

### What changed

- `_normalize_protein_ids` now converts every non-missing ID to text before trimming. For example, `123` and `" 123 "` both normalize to `"123"`.
- Added a direct behavioral test that verifies numeric IDs share the same positive relation as their trimmed text representation.
- The deferred empty-batch Minor was intentionally not changed in this round.

### TDD evidence

#### RED

Command:

```powershell
rtk proxy C:\Users\87279\miniconda3\envs\psl\python.exe -m pytest tests\test_protein_aware_isc.py -q
```

Result: `1 failed, 7 passed in 2.83s`. The new numeric-ID test failed for the expected reason: `ValueError: Protein ID at index 0 must be a string`.

#### GREEN

Command:

```powershell
rtk proxy C:\Users\87279\miniconda3\envs\psl\python.exe -m pytest tests\test_protein_aware_isc.py -q
```

Result: `8 passed in 2.75s`.

### Full verification

Command:

```powershell
rtk proxy C:\Users\87279\miniconda3\envs\psl\python.exe -m pytest -q
```

Result: `101 passed, 23 subtests passed in 12.35s`.

### Fix self-review

- Missing (`None` or `NaN`) and blank values remain explicit `ValueError` cases.
- The change is limited to normalization; loss reduction and live training/validation loops remain unchanged.

## Fix round 2 — non-native scalar NaN IDs

### What changed

- Added a scalar `math.isnan` check before text conversion, so native and NumPy floating-point NaN IDs are both explicit missing-value errors.
- Added direct coverage for `np.float32("nan")` while retaining numeric-ID acceptance.
- The deferred empty-collection Minor remains intentionally unchanged.

### TDD evidence

#### RED

Command:

```powershell
rtk proxy C:\Users\87279\miniconda3\envs\psl\python.exe -m pytest tests\test_protein_aware_isc.py -q
```

Result: `1 failed, 8 passed in 2.81s`. The new NumPy scalar-NaN test failed for the expected reason: `Failed: DID NOT RAISE ValueError`.

#### GREEN

Command:

```powershell
rtk proxy C:\Users\87279\miniconda3\envs\psl\python.exe -m pytest tests\test_protein_aware_isc.py -q
```

Result: `9 passed in 2.82s`.

### Full verification

Command:

```powershell
rtk proxy C:\Users\87279\miniconda3\envs\psl\python.exe -m pytest -q
```

Result: `102 passed, 23 subtests passed in 12.27s`.

### Fix self-review

- `math.isnan` accepts NumPy floating scalar values and leaves valid numeric identifiers such as `123` convertible to text.
- The change precedes `str(...)`, preventing missing numeric sentinel values from becoming the literal ID `"nan"`.
