# Official HQ Manifests Exactly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish HQ train/test URL manifests whose rows and order come only from the pinned Vislocas `data_train.csv` and `data_test.csv`, failing the whole publication when any official row is unusable.

**Architecture:** Add a focused `official_hq_manifests.py` command that consumes the verified-source API committed by ticket 01. It validates both official frames entirely in memory, resolves reviewed sequences through an injectable boundary, derives download fields without filtering or reclassification, and atomically replaces the two manifests plus an occupied-source-row registry only after every row passes. This separate module avoids depending on or staging the pre-existing uncommitted generator rewrite in `filter_quality_urls.py`.

**Tech Stack:** Python 3.11+, pandas, requests, standard-library `argparse`, `csv`, `hashlib`, `json`, `pathlib`, `shutil`, `tempfile`, and `unittest`.

**BASE_SHA:** `316613d3e656fae1096b160df64aa005c8001e82`

## Global Constraints

- Reuse ticket 01's pinned Zenodo record `10632698`, exact three-file catalog, MD5 verification, official source-row validation, and atomic cache behavior.
- Build HQ train only from `data_train.csv` and HQ test only from `data_test.csv`; `normalLabeled.csv` is used only by ticket 01 to validate exact source-row identity.
- Preserve every official row, its multiplicity, its original values, and its relative order. Never classify, deduplicate, demote, move, or silently skip an official row.
- Normalize only leading/trailing whitespace when comparing `Protein Id`; reject train/test overlap on that key alone.
- Do not inspect or constrain URL, image name, filename, or image bytes when deciding split ownership.
- Require one unique, nonblank reviewed sequence per normalized official `Protein Id`.
- Treat blank or malformed download-required fields as fatal official-row failures.
- Validate all official rows before publishing; a failure must not replace either existing formal HQ manifest or the occupied-row registry.
- Record official occupancy by normalized source-row identity, not URL, so later supplemental processing can exclude exactly those source rows.
- Keep pre-existing worktree changes untouched and stage only files created or edited for ticket 02.
- Automated tests use local CSV fixtures and injected sequence/HTTP boundaries; do not access Zenodo, UniProt, or Protein Atlas.

## File Structure

- Create `dataset/download/official_hq_manifests.py`: official-only manifest assembly, reviewed-sequence resolver, fatal-row reporting, occupied-row registry, atomic bundle publication, and CLI.
- Create `dataset/download/test_official_hq_manifests.py`: CLI fixture tests, sequence-resolution tests, fatal validation tests, and publication rollback tests.
- Modify `.scratch/official-hq-rgb-dataset/issues/02-publish-official-hq-manifests-exactly.md`: mark only ticket 02 done after all verification and review gates pass.
- Do not modify or stage the already-dirty `dataset/download/filter_quality_urls.py` and `dataset/download/test_filter_quality_urls.py`; import only ticket 01 contracts present at `BASE_SHA`.

---

### Task 1: Official-row assembly and successful CLI publication

**Files:**
- Create: `dataset/download/test_official_hq_manifests.py`
- Create: `dataset/download/official_hq_manifests.py`

**Interfaces:**
- Consumes: ticket 01's `prepare_verified_sources`, `validate_official_rows`, `SOURCE_ROW_ID`, `SOURCE_URLS`, and `SOURCE_MD5`; verified `data_train.csv` and `data_test.csv`; an injectable `sequence_resolver(protein_ids, cache_path) -> tuple[dict[str, str], set[str]]`.
- Produces: `assemble_official_hq(train, test, sequences) -> dict[str, DataFrame]`, `build_image_fields(antibody_id, relative_url) -> tuple[str, str, str]`, and CLI files `HQ_train_img_URL.csv`, `HQ_test_img_URL.csv`, `official_hq_occupied_rows.csv`, `manifest_failures.csv`, `manifest_generation_report.json`, and `source_validation_report.json`.

- [ ] **Step 1: Add a failing CLI fixture test for exact official membership and order**

Create complete small source rows with source-row IDs `[10, 20, 30, 40]`; set official train order to `[20, 10]`, official test order to `[40, 30]`, and leave one extra supplemental row in `normalLabeled.csv`. Give different proteins the same relative URL to prove URL equality does not affect ownership. Write all three fixtures to a temporary cache, calculate fixture MD5 values, inject a resolver returning one sequence per official protein, and assert:

```python
exit_code = main(
    ["--cache-dir", str(cache_dir), "--output-dir", str(output_dir)],
    source_urls=fixture_urls,
    source_md5=fixture_md5,
    sequence_resolver=fake_resolver,
)
self.assertEqual(exit_code, 0)
self.assertEqual(
    pd.read_csv(output_dir / "HQ_train_img_URL.csv")[SOURCE_ROW_ID].tolist(),
    [20, 10],
)
self.assertEqual(
    pd.read_csv(output_dir / "HQ_test_img_URL.csv")[SOURCE_ROW_ID].tolist(),
    [40, 30],
)
```

Also assert both output row counts equal their official inputs, the extra supplemental row is absent, shared URLs are retained, derived sequences and image URLs are present, and no HTTP mock was called.

- [ ] **Step 2: Run the CLI fixture test and verify RED**

Run:

```powershell
python -m unittest dataset.download.test_official_hq_manifests.OfficialHqCliTest.test_cli_publishes_only_official_rows_in_official_order -v
```

Expected: import failure because `official_hq_manifests.py` does not exist.

- [ ] **Step 3: Implement official manifest constants and pure row assembly**

Define `SOURCE_COLUMNS` as the existing official URL-manifest columns through `vesicles`, then append exactly:

```python
OUTPUT_COLUMNS = [
    *SOURCE_COLUMNS,
    "Sequence",
    "NonZeroDigits",
    "Image Name",
    "Modified URL",
]
```

`assemble_official_hq` must copy each input frame, normalize only `Protein Id`, require all `SOURCE_COLUMNS`, reject blank normalized IDs, require a nonblank sequence mapping entry, derive image fields row-by-row without selecting or sorting rows, and return `{"HQ_train": ..., "HQ_test": ...}` with `OUTPUT_COLUMNS` in fixed order. It must check normalized train/test `Protein Id` sets before deriving URL fields and raise with the sorted overlap. It must not receive `normalLabeled.csv` and must not call any quality classifier.

- [ ] **Step 4: Implement the initial CLI orchestration**

Add arguments `--cache-dir` and `--output-dir`. Call `prepare_verified_sources(cache_dir, source_urls, source_md5)` first, load the two official CSVs directly from `cache_dir`, call `validate_official_rows` against `normalLabeled.csv`, resolve the union of nonblank normalized official protein IDs, assemble both frames in memory, and publish outputs only after assembly succeeds. Accept `sequence_resolver` as a keyword-only dependency of `main`; default it to the production resolver added in Task 2.

- [ ] **Step 5: Run the focused CLI test and verify GREEN**

Run the Step 2 command. Expected: one test passes with final `OK`.

- [ ] **Step 6: Run ticket 01 dependency tests again**

Run the six pinned-source/cache/source-row tests used at the dependency gate. Expected: all six pass; the new module must not alter ticket 01 behavior.

### Task 2: Unique reviewed-sequence boundary and fatal row reports

**Files:**
- Modify: `dataset/download/test_official_hq_manifests.py`
- Modify: `dataset/download/official_hq_manifests.py`

**Interfaces:**
- Produces: `parse_uniprot_results(payload) -> tuple[dict[str, str], set[str]]`, `fetch_reviewed_sequences(protein_ids, cache_path) -> tuple[dict[str, str], set[str]]`, and `collect_official_failures(train, test, sequences, unresolved) -> DataFrame` using `MANIFEST_FAILURE_COLUMNS = ["stage", "tier", "split", "source_line", "source_row", "Protein Id", "URL", "reason"]`.

- [ ] **Step 1: Add failing sequence uniqueness tests**

Test that two identical reviewed results collapse to one sequence, two distinct nonblank sequences for the same source ID mark that ID unresolved, and blank sequences do not resolve. Test a cache containing conflicting sequences for one protein and a unique sequence for another; the conflicting protein must remain unresolved and the HTTP client must not be called for it.

- [ ] **Step 2: Run sequence tests and verify RED**

Run:

```powershell
python -m unittest dataset.download.test_official_hq_manifests.SequenceResolutionTest -v
```

Expected: failures because the parser and resolver are absent.

- [ ] **Step 3: Implement reviewed-only UniProt resolution**

Use UniProt ID mapping with `from=Ensembl` and `to=UniProtKB-Swiss-Prot`, poll the job with a bounded deadline, request JSON pages containing `sequence`, and merge all results by source ID. Strip returned sequence text; accept an ID only when its set contains exactly one nonblank sequence. Read and atomically rewrite `uniprot_sequences.csv`; duplicate cached values make an ID unresolved and must not be silently overwritten. Return every requested-but-not-uniquely-resolved ID in the unresolved set.

- [ ] **Step 4: Add failing CLI tests for normalized overlap, missing/ambiguous sequence, and malformed image fields**

Use independent CLI fixtures to assert:

- train `"P1"` and test `" P1 "` fail before sequence resolution;
- an unresolved official protein produces one failure row for every corresponding official source line;
- a blank returned sequence is fatal even if it is present in the resolver mapping;
- antibody IDs without digits, blank URLs, URL paths without an image name, blank `locations`, and blank label values are reported and no formal manifest is newly created.

For every case, assert `main` returns `1`, `manifest_generation_report.json` has `status="error"` and `published=false`, `manifest_failures.csv` identifies `tier=HQ`, split, 1-based CSV source line (header included), exact source-row identity, normalized Protein Id, URL, stage, and reason.

- [ ] **Step 5: Run fatal validation tests and verify RED**

Run:

```powershell
python -m unittest dataset.download.test_official_hq_manifests.OfficialHqFailureTest -v
```

Expected: assertions fail because fatal failures are not yet converted into row-level reports.

- [ ] **Step 6: Implement exhaustive pre-publication validation and reporting**

Validate both frames without filtering. Build failures in original split/row order, using `source_line = position + 2`. Sequence failure stage is `sequence`; missing/malformed required fields use `required_field` or `image_fields`; normalized Protein Id overlap uses `split_overlap` and reports both participating sides. If any failure exists, atomically write the failure CSV and JSON report, leave all existing formal HQ artifacts untouched, print the report to stderr, and return `1`.

- [ ] **Step 7: Run sequence and fatal validation tests and verify GREEN**

Run both Step 2 and Step 5 commands. Expected: all tests pass.

### Task 3: Exact occupied-row registry, historical-file independence, and atomic bundle rollback

**Files:**
- Modify: `dataset/download/test_official_hq_manifests.py`
- Modify: `dataset/download/official_hq_manifests.py`

**Interfaces:**
- Produces: `build_occupied_rows(train, test) -> DataFrame` and `publish_official_bundle(outputs, occupied_rows, output_dir, replace=os.replace) -> None`.

- [ ] **Step 1: Add a failing occupied-row and stale-history CLI test**

Precreate poisoned `train_img_URL.csv`, `test_img_URL.csv`, `HQ_only_generated.csv`, and `HQ_comparison_report.txt` in the output directory. Run the successful CLI fixture and assert output bytes/rows match a clean run, none of those files is read or modified, and:

```python
occupied = pd.read_csv(
    output_dir / "official_hq_occupied_rows.csv", dtype=str
)
self.assertEqual(occupied["split"].tolist(), ["train", "train", "test", "test"])
self.assertEqual(occupied["source_row"].tolist(), ["20", "10", "40", "30"])
```

Add another legal fixture where different official proteins share an identical URL across train/test and assert successful publication.

- [ ] **Step 2: Run stale-history tests and verify RED**

Run the two focused tests. Expected: failure until the occupied registry and explicit no-history CLI contract exist.

- [ ] **Step 3: Add a failing publication rollback test**

Seed all three formal artifacts with old bytes. Inject a `replace` function that succeeds for the first staged artifact and fails for the second. Assert `publish_official_bundle` raises, restores all three original byte strings, and leaves no `.part`, staging, or backup artifact.

- [ ] **Step 4: Implement the three-artifact transaction**

Write both manifests and the occupied registry into a unique staging directory under `output_dir`. Move existing destinations to unique backups, replace all three destinations in fixed order, and on any error remove newly published destinations and restore every backup. Always remove staging and obsolete backup paths. The registry columns are exactly `split`, `source_position`, `source_line`, and `source_row`; `source_row` uses the same numeric/string canonicalization as ticket 01, while the manifests retain the official row values.

- [ ] **Step 5: Run Task 3 tests and verify GREEN**

Run the occupied/stale-history tests and rollback test. Expected: all pass.

### Task 4: CLI contract, requirement audit, review, and ticket completion

**Files:**
- Verify: `dataset/download/official_hq_manifests.py`
- Verify: `dataset/download/test_official_hq_manifests.py`
- Modify: `.scratch/official-hq-rgb-dataset/issues/02-publish-official-hq-manifests-exactly.md`

- [ ] **Step 1: Run the complete new test module**

Run:

```powershell
python -m unittest dataset.download.test_official_hq_manifests -v
```

Expected: every new unit and CLI fixture test passes.

- [ ] **Step 2: Run the full dataset-download and repository suites**

Run:

```powershell
python -m unittest discover -s dataset/download -p "test_*.py" -v
python -m unittest discover -s tests -v
```

Record exact pass/failure counts. Investigate failures caused by ticket 02; report unrelated pre-existing failures without weakening the ticket requirements.

- [ ] **Step 3: Compile and inspect the CLI without network**

Run:

```powershell
python -m py_compile dataset/download/official_hq_manifests.py dataset/download/test_official_hq_manifests.py
python dataset/download/official_hq_manifests.py --help
```

Expected: compilation exits 0; help lists only `--cache-dir` and `--output-dir`; importing the module performs no I/O.

- [ ] **Step 4: Inspect exact Git scope and commit ticket files only**

Run `git diff --check`, inspect the three ticket files, stage them with explicit paths, and inspect `git diff --cached --check` plus `git diff --cached`. Commit with:

```powershell
git commit -m "feat: publish official HQ manifests exactly"
```

Do not stage the pre-existing changes in `filter_quality_urls.py`, `test_filter_quality_urls.py`, or unrelated files.

- [ ] **Step 5: Run the `code-review` skill from fixed point `BASE_SHA`**

Review standards and spec compliance against ticket 02 plus the approved design. Fix every blocking finding with TDD, rerun the relevant focused and full suites, and commit review fixes separately.

- [ ] **Step 6: Mark only ticket 02 done**

After review and fresh verification succeed, change `Status` to `done` and tick all eight ticket 02 acceptance boxes. Do not edit tickets 03–07.

- [ ] **Step 7: Final verification and handoff**

Run the complete new test module, dependency tests, applicable full suites, compile checks, CLI help, `git diff --check`, `git status --short`, and `git log -3 --oneline`. Report exact evidence, commit SHA(s), untouched pre-existing worktree changes, and any remaining risk. Do not run real production downloads or push.
