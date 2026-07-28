# End-to-End Dataset Pipeline Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the complete pinned-source-to-six-RGB-dataset workflow through the two public command-line entry points with a small deterministic fixture and operator-usable failure evidence.

**Architecture:** Add one self-contained acceptance-test module that calls `supplemental_quality_manifests.main(argv, ...)` and `download.main(argv, ...)`, while replacing only the external source, sequence, and HTTP boundaries. Use real temporary directories and inspect only public artifacts: six URL manifests, six training CSVs, image files, failure CSVs, JSON audit reports, stdout/stderr, and process exit status. Keep production code unchanged unless a failing acceptance test demonstrates a contract gap.

**Tech Stack:** Python 3.11, standard-library `unittest`, `tempfile`, `subprocess`, `hashlib`, `io`, `json`, pandas 2.3, Pillow, and the existing CLI injection seams.

## Global Constraints

- Exercise the two formal entry points; do not call private generation or download helpers to perform the pipeline.
- Use only temporary fixture sources and controlled sequence/HTTP boundaries; never request the real 88 MB table or production image corpus.
- Preserve official HQ membership and order exactly.
- Treat normalized `Protein Id` as the only train/test leakage key; retain a cross-protein shared URL across splits.
- Cover official HQ, supplemental-HQ-to-MQ demotion, MQ, LQ, known and unknown proteins, RGB JPEG, non-RGB, alpha transparency, all-white content, HTTP failure, and corrupt image content.
- Require all successful images to reopen as JPEG, RGB, and exactly three bands.
- Require all six training CSVs to use the exact nine-column contract with `Protein Id` last.
- Run the same source fixture and seed twice and compare manifest bytes, formal CSV bytes, filenames, and audit statistics.
- Preserve pre-existing dirty worktree changes and stage only ticket-07-owned files.
- Do not install or upgrade dependencies; record the existing downstream `torch` environment failure if it remains.

---

### Task 1: Full two-CLI acceptance fixture

**Files:**
- Create: `dataset/download/test_dataset_pipeline_e2e.py`

**Interfaces:**
- Consumes: `supplemental_quality_manifests.main(argv, source_urls=..., source_md5=..., sequence_resolver=...)` and `download.main(argv, http_get=...)`.
- Produces: `DatasetPipelineEndToEndTest.test_two_cli_pipeline_is_deterministic_and_contract_complete`.

- [ ] **Step 1: Build independent fixture primitives**

Create literal source rows containing every required source column. The official train rows use source IDs `[20, 10]`, the official test row uses `[30]`, supplemental known rows inherit both official sides, and ten unknown proteins make the 10% deterministic split observable. Give two different known proteins on opposite sides the same antibody/image pair so their generated `Modified URL` values are equal.

Use a resolver whose complete behavior is:

```python
def fixture_sequence_resolver(protein_ids, _cache_path):
    normalized = {str(value).strip() for value in protein_ids}
    return ({protein_id: f"SEQUENCE_{protein_id}" for protein_id in normalized}, set())
```

Build controlled payloads for RGB JPEG, grayscale PNG, RGBA PNG with transparent pixels, an all-white PNG, invalid bytes, and HTTP 503.

- [ ] **Step 2: Add the end-to-end test through both public mains**

For each of two independent temporary run roots, invoke:

```python
manifest_code = manifests.main(
    ["--cache-dir", str(cache_dir), "--output-dir", str(manifest_dir),
     "--seed", "73"],
    source_urls=fixture_urls,
    source_md5=fixture_md5,
    sequence_resolver=fixture_sequence_resolver,
)
download_code = downloader.main(
    ["--manifest-dir", str(manifest_dir), "--output-dir", str(output_dir),
     "--workers", "3"],
    http_get=controlled_http_get,
)
```

Do not invoke `generate_quality_manifests`, `run_download_pipeline`, or a private helper to perform either stage.

- [ ] **Step 3: Assert public artifacts and policies**

Assert literal expectations:

```python
self.assertEqual(generated_hq_train["Unnamed: 0"].tolist(), [20, 10])
self.assertEqual(generated_hq_test["Unnamed: 0"].tolist(), [30])
self.assertEqual(final_frame.columns.tolist(), EXPECTED_FINAL_COLUMNS)
self.assertEqual(train_protein_ids & test_protein_ids, set())
```

Also assert that the supplemental HQ row is in MQ, the shared URL remains once on each split for different proteins, blank/HTTP/corrupt MQ/LQ rows are absent from formal CSVs and present in `download_failures.csv`, every successful file is listed exactly once, and Pillow reopens every success as JPEG/RGB/three-band.

- [ ] **Step 4: Assert repeatability**

Compare the two runs' six manifest byte sequences, six formal CSV byte sequences, per-dataset ordered filename lists, `manifest_generation_report.json`, and `download_audit_report.json`. HTTP completion order is irrelevant; formal order must remain source order.

- [ ] **Step 5: Run the focused test**

Run:

```powershell
rtk python -m unittest dataset.download.test_dataset_pipeline_e2e.DatasetPipelineEndToEndTest.test_two_cli_pipeline_is_deterministic_and_contract_complete -v
```

Expected: `OK`; if it fails, the failure must identify a real public contract gap before production code is changed.

---

### Task 2: CLI help and failure diagnostics

**Files:**
- Modify: `dataset/download/test_dataset_pipeline_e2e.py`
- Modify only if a failing test proves necessary: `dataset/download/supplemental_quality_manifests.py`
- Modify only if a failing test proves necessary: `dataset/download/download.py`

**Interfaces:**
- Consumes: both script files as actual subprocess entry points for `--help`, manifest `main` for an injected source failure, and downloader script for a missing-manifest failure.
- Produces: tests that prove help returns zero and failures return nonzero while writing a deterministic report under the requested output directory.

- [ ] **Step 1: Add help contract tests**

Run each script with `sys.executable` and `--help`. Assert return code `0`, empty network activity by construction, and the documented arguments:

```python
manifest_help = {"--output-dir", "--cache-dir", "--seed"}
download_help = {"--manifest-dir", "--output-dir", "--workers"}
```

- [ ] **Step 2: Add manifest failure diagnostics**

Use fixture official train/test frames with the same trimmed Protein Id on both sides. Invoke the public manifest `main`; assert return code `1`, stderr JSON with `status=error`, and `manifest_generation_report.json` plus `manifest_failures.csv` in the requested output directory.

- [ ] **Step 3: Add downloader process failure diagnostics**

Run the downloader script against a temporary manifest directory missing one formal manifest. Assert a nonzero process return code and inspect `<output-dir>/download_audit_report.json` for `status=error`, `published=false`, and an error message naming the missing artifact. No HTTP boundary is reached.

- [ ] **Step 4: Run the complete acceptance module**

```powershell
rtk python -m unittest dataset.download.test_dataset_pipeline_e2e -v
```

Expected: every acceptance test passes. If production error handling changes, first observe the relevant test fail for that exact missing behavior, then implement only the smallest public-boundary fix.

---

### Task 3: Deprecated-contract and repository audit

**Files:**
- Audit and minimally update when required: `dataset/download/test_filter_quality_urls.py`
- Audit and minimally update when required: `dataset/download/filter_quality_urls.py`
- Verify: `dataset/download/official_hq_manifests.py`
- Verify: `dataset/download/supplemental_quality_manifests.py`
- Verify: `dataset/download/download.py`

**Interfaces:**
- Consumes: the approved specification and all dataset-download tests.
- Produces: no active test expectation that prefers one quality tier for a shared URL, requires URL cross-tier exclusivity, or reconstructs official HQ from the supplemental source.

- [ ] **Step 1: Search only source and test files for obsolete contracts**

```powershell
rtk rg -n -g "*.py" "resolve_url_quality_conflicts|overlapping URLs|cross_tier_conflict|duplicate_url_rows_removed|compare_hq|existing HQ" dataset/download
rtk rg -n -g "*.py" "TODO|TBD|FIXME|HACK" dataset/download
```

Classify every match. Delete or replace only tests that require superseded behavior; do not weaken the formal shared-URL assertions in the official/supplemental/downloader tests.

- [ ] **Step 2: Run all dataset tests after cleanup**

```powershell
rtk python -m unittest discover -s dataset/download -p "test_*.py" -v
```

Expected: all discovered tests pass, including the new acceptance module.

- [ ] **Step 3: Compile all dataset-download Python files**

```powershell
rtk python -m compileall -q dataset/download
```

Expected: exit code `0` with no syntax error.

---

### Task 4: Evidence, fixed-base review, and ticket closure

**Files:**
- Modify: `.scratch/official-hq-rgb-dataset/issues/07-verify-end-to-end-dataset-pipeline.md`
- Verify: all ticket-owned files since base `085c729687c157b0c5a373d5f5284a4a731b40ed`

**Interfaces:**
- Consumes: fresh test, compile, CLI, audit, Git-scope, and review evidence.
- Produces: ticket 07 marked `done`, scoped commits, and a fixed-base standards/spec review with no blocking findings.

- [ ] **Step 1: Run final applicable verification**

```powershell
rtk python -m unittest discover -s dataset/download -p "test_*.py" -v
rtk python -m unittest discover -s tests -v
rtk python -m compileall -q dataset/download
rtk python dataset/download/supplemental_quality_manifests.py --help
rtk python dataset/download/download.py --help
rtk git -c safe.directory=E:/work/ProLoc-IHCS diff --check
```

Record exact counts and any environment-only downstream failure. Do not install `torch` or alter training code.

- [ ] **Step 2: Mark ticket 07 complete only after evidence is fresh**

Change only ticket 07 from `ready-for-agent` to `done` and its eight checklist markers from `[ ]` to `[x]` once every ticket-scoped requirement is satisfied. If the downstream environment failure prevents an honest checkbox, leave the affected item open and report it.

- [ ] **Step 3: Stage and commit ticket-owned changes**

```powershell
rtk git -c safe.directory=E:/work/ProLoc-IHCS add -- dataset/download/test_dataset_pipeline_e2e.py docs/superpowers/plans/2026-07-28-end-to-end-dataset-pipeline-verification.md .scratch/official-hq-rgb-dataset/issues/07-verify-end-to-end-dataset-pipeline.md
rtk git -c safe.directory=E:/work/ProLoc-IHCS diff --cached --check
rtk git -c safe.directory=E:/work/ProLoc-IHCS commit -m "test: verify dataset pipeline end to end"
```

Add a production module to the staged list only if Task 2 proved and fixed a public contract gap. Do not stage unrelated dirty paths.

- [ ] **Step 4: Review from the fixed base**

Invoke the repository `code-review` skill with fixed point `085c729687c157b0c5a373d5f5284a4a731b40ed`, ticket 07, and the approved specification. Repair every blocking ticket-scoped finding, rerun Task 4 Step 1, and commit any review fix separately.

- [ ] **Step 5: Inspect final scope**

```powershell
rtk git -c safe.directory=E:/work/ProLoc-IHCS status --short
rtk git -c safe.directory=E:/work/ProLoc-IHCS log -5 --oneline
```

Expected: no ticket-owned file remains uncommitted, every pre-existing dirty path remains otherwise preserved, and no push or production download occurred.
