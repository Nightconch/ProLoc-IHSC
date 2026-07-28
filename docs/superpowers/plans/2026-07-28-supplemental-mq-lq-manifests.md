# Supplemental MQ/LQ Manifests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish deterministic HQ/MQ/LQ train/test URL manifests in which official HQ rows remain exact, supplemental rows retain shared URLs, and normalized Protein Id values never cross train/test.

**Architecture:** Add an isolated manifest-orchestration module that reuses the pinned-source boundary from `filter_quality_urls.py` and the official-HQ assembly and sequence boundary from `official_hq_manifests.py`. The new module classifies only non-official rows, creates one deterministic protein split map before row-level supplemental failures are removed, assembles all six frames in memory, checks global Protein Id disjointness, and atomically replaces the six formal CSVs. This avoids staging the large pre-existing uncommitted changes in the legacy generator.

**Tech Stack:** Python 3.11, pandas 2.3, requests 2.32 through existing boundaries, standard-library `unittest`, `random`, `heapq`, `json`, `pathlib`, and atomic filesystem replacement.

## Global Constraints

- Use only Zenodo record `10632698` inputs accepted by the existing pinned-source validator.
- Build `HQ_train` only from `data_train.csv` and `HQ_test` only from `data_test.csv`; retain official membership, values, and relative order.
- Use `normalLabeled.csv` as the only supplemental source and exclude only exact official source-row identities returned by `validate_official_rows`.
- Apply the existing five-label eligibility and intensity/quantity rules; publish supplemental HQ-classified rows as MQ and retain MQ/LQ classifications.
- Strip leading/trailing whitespace only for Protein Id comparisons, sequence lookup, and split assignment; preserve official source values in formal rows.
- Report and skip supplemental rows with blank/missing Protein Id, unresolved/ambiguous reviewed sequence, or invalid download fields.
- Assign known proteins to their official side and assign unknown proteins once, using a fixed seed and deterministic approximate multilabel stratification targeting 10% unknown proteins in test.
- Compute the split map before supplemental sequence or image-field failures so failures never repartition proteins.
- Never deduplicate, remove, or bind records by URL, image name, or pixel hash.
- Preserve source-relative row order within every manifest.
- Assert normalized Protein Id train/test disjointness across all six frames before creating any formal staging file.
- Publish exactly `HQ_train_img_URL.csv`, `HQ_test_img_URL.csv`, `MQ_train_img_URL.csv`, `MQ_test_img_URL.csv`, `LQ_train_img_URL.csv`, and `LQ_test_img_URL.csv` as one rollback-capable transaction.
- Automated tests use local fixtures and an injected sequence resolver; they never access the real upstream record or download images.
- Preserve all baseline changes that existed at `87aac9f3a3fb85c9ee645a724236b86c6e4322ef`; stage only files created or intentionally modified for ticket 03.

## File Structure

- Create `dataset/download/supplemental_quality_manifests.py`: supplemental quality rules, deterministic protein assignment, six-frame assembly, failure reporting, atomic publication, and CLI.
- Create `dataset/download/test_supplemental_quality_manifests.py`: pure-rule tests plus CLI fixture and atomic rollback coverage.
- Modify `.scratch/official-hq-rgb-dataset/issues/03-publish-supplemental-mq-lq-without-protein-leakage.md` only after every acceptance check passes.
- Do not modify or stage the already-dirty `dataset/download/filter_quality_urls.py` or `dataset/download/test_filter_quality_urls.py`.

---

### Task 1: Supplemental row preparation and audit failures

**Files:**
- Create: `dataset/download/supplemental_quality_manifests.py`
- Create: `dataset/download/test_supplemental_quality_manifests.py`

**Interfaces:**
- Consumes: `normalLabeled.csv` as a `pandas.DataFrame` and normalized official source-row identities as `set[str]`.
- Produces: `classify_quality(intensity, quantity) -> str | None` and `prepare_supplemental_rows(source, official_row_ids) -> tuple[pd.DataFrame, list[dict], dict]`.
- Prepared rows carry a private `__source_line` column for reporting, a normalized `__protein_id` column for mapping, and a public `Quality` value of only `MQ` or `LQ`.

- [ ] **Step 1: Write failing rule and preparation tests**

Create fixture rows with all `SOURCE_COLUMNS` and five download labels, then add tests equivalent to:

```python
def test_prepare_excludes_exact_official_rows_demotes_hq_and_reports_blank_ids(self):
    source = pd.DataFrame([
        quality_row(10, "P_OFFICIAL", "strong", ">75%", "official.jpg"),
        quality_row(11, " P_MQ ", "strong", ">75%", "mq.jpg"),
        quality_row(12, "P_LQ", "weak", ">75%", "lq.jpg"),
        quality_row(13, "   ", "moderate", ">75%", "blank.jpg"),
    ])

    prepared, failures, stats = prepare_supplemental_rows(source, {"10"})

    self.assertEqual(prepared[SOURCE_ROW_ID].tolist(), [11, 12])
    self.assertEqual(prepared["Quality"].tolist(), ["MQ", "LQ"])
    self.assertEqual(prepared["__protein_id"].tolist(), ["P_MQ", "P_LQ"])
    self.assertEqual(stats["supplemental_hq_demoted"], 1)
    self.assertEqual([(row["source_row"], row["stage"]) for row in failures], [("13", "protein_id")])
```

Also assert that `("moderate", ">75%")` and `("strong", "75%-25%")` are MQ; weak/moderate lower pairs are LQ; any `("strong", ">75%")` wins as HQ before demotion; mismatched semicolon lists are reported as `stage=quality`; non-enhanced, blank-location, nonnumeric-label, and all-zero-label rows are not supplemental candidates.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
rtk python -m unittest dataset.download.test_supplemental_quality_manifests.SupplementalPreparationTest -v
```

Expected: import failure because `supplemental_quality_manifests.py` does not exist.

- [ ] **Step 3: Implement the quality rules and preparation boundary**

Use these public constants and result shape:

```python
LABEL_COLUMNS = list(DOWNLOAD_LABEL_COLUMNS)
MANIFEST_FAILURE_COLUMNS = [
    "stage", "tier", "split", "source_line", "source_row",
    "Protein Id", "URL", "reason",
]

def classify_quality(intensity, quantity):
    intensities = split_quality_values(intensity)
    quantities = split_quality_values(quantity)
    if len(intensities) != len(quantities):
        raise ValueError("intensity and quantity must have the same number of values")
    result = None
    for pair in zip(intensities, quantities):
        if pair == ("strong", ">75%"):
            return "HQ"
        if pair in {("moderate", ">75%"), ("strong", "75%-25%")}:
            result = "MQ"
        elif result is None and pair in {
            ("weak", ">75%"),
            ("moderate", "75%-25%"),
            ("weak", "75%-25%"),
        }:
            result = "LQ"
    return result
```

`prepare_supplemental_rows` must add one-based CSV data positions as `__source_line = position + 2`, normalize source-row identities with `_source_row_identity`, exclude official identities before eligibility/classification, preserve DataFrame order, report classified rows whose normalized Protein Id is empty, set `__protein_id` for the remaining rows, count HQ classifications, and replace only those HQ values with MQ.

- [ ] **Step 4: Run Task 1 tests and verify GREEN**

Run the command from Step 2. Expected: every `SupplementalPreparationTest` test passes and the final output is `OK`.

- [ ] **Step 5: Commit the preparation slice**

```powershell
rtk git -c safe.directory=E:/work/ProLoc-IHCS add -- dataset/download/supplemental_quality_manifests.py dataset/download/test_supplemental_quality_manifests.py docs/superpowers/plans/2026-07-28-supplemental-mq-lq-manifests.md
rtk git -c safe.directory=E:/work/ProLoc-IHCS diff --cached --check
rtk git -c safe.directory=E:/work/ProLoc-IHCS commit -m "feat: prepare supplemental quality rows"
```

---

### Task 2: One deterministic split map and six-frame invariants

**Files:**
- Modify: `dataset/download/supplemental_quality_manifests.py`
- Modify: `dataset/download/test_supplemental_quality_manifests.py`

**Interfaces:**
- Consumes: prepared supplemental rows and the official train/test frames.
- Produces: `assign_protein_splits(supplemental, official_train, official_test, seed) -> tuple[dict[str, str], dict]`, `assemble_quality_outputs(official_train, official_test, supplemental, split_mapping, sequences) -> dict[str, pd.DataFrame]`, and `assert_protein_disjoint(outputs) -> None`.

- [ ] **Step 1: Add failing known/unknown assignment tests**

Build a prepared fixture containing known official proteins plus ten unknown proteins with varied five-label vectors. Assert:

```python
mapping_a, stats_a = assign_protein_splits(rows, official_train, official_test, seed=73)
mapping_b, stats_b = assign_protein_splits(rows, official_train, official_test, seed=73)
self.assertEqual(mapping_a, mapping_b)
self.assertEqual(stats_a, stats_b)
self.assertEqual(mapping_a["P_KNOWN_TRAIN"], "train")
self.assertEqual(mapping_a["P_KNOWN_TEST"], "test")
self.assertEqual(sum(mapping_a[p] == "test" for p in unknown_ids), 1)
```

Include two rows whose raw values are `P_SHARED` and ` P_SHARED ` and assert they receive one normalized mapping entry.

- [ ] **Step 2: Run the assignment tests and verify RED**

```powershell
rtk python -m unittest dataset.download.test_supplemental_quality_manifests.ProteinAssignmentTest -v
```

Expected: attribute/import failure for `assign_protein_splits`.

- [ ] **Step 3: Implement deterministic approximate multilabel stratification**

Normalize official IDs and reject an official train/test overlap. Aggregate the prepared rows by `__protein_id` with label-wise maximum and convert values to booleans. Sort unknown IDs before consuming randomness. Set `test_count = min(len(unknown), max(1, round(len(unknown) * 0.1)))` when unknown IDs exist. Use `random.Random(seed)` only to generate stable tie values, then greedily choose the candidate whose addition minimizes squared distance from `protein_labels.sum() * 0.1`; break equal scores by the generated tie value and Protein Id. Return mapping statistics containing `known_train_proteins`, `known_test_proteins`, `unknown_proteins`, and `unknown_test_proteins`.

- [ ] **Step 4: Add failing six-frame and shared-URL tests**

Use official train source-row order `[20, 10]`. Give an MQ row and an LQ row different Protein Id values but the same antibody number/image name, then assert both survive and have the same `Modified URL`. Add an invariant test where `P1` and ` P1 ` appear on opposite sides:

```python
with self.assertRaisesRegex(AssertionError, "Protein Id overlap.*P1"):
    assert_protein_disjoint(outputs_with_whitespace_overlap)
```

- [ ] **Step 5: Implement row assembly and the global invariant**

`assemble_quality_outputs` must use `assemble_official_hq` for HQ, build supplemental derived fields using normalized IDs for sequence lookup without rewriting the public Protein Id, filter supplemental rows by their single mapping, and preserve source order. It returns exactly these keys:

```python
REQUIRED_OUTPUTS = {
    "HQ_train", "HQ_test", "MQ_train", "MQ_test", "LQ_train", "LQ_test"
}
```

`assert_protein_disjoint` unions normalized, nonblank Protein Id values from every `*_train` frame and every `*_test` frame, sorts the intersection, and raises before returning if it is nonempty. Do not inspect URL values.

- [ ] **Step 6: Run Task 2 tests and verify GREEN**

```powershell
rtk python -m unittest dataset.download.test_supplemental_quality_manifests.ProteinAssignmentTest dataset.download.test_supplemental_quality_manifests.ManifestAssemblyTest -v
```

Expected: all assignment, row-order, shared-URL, and normalized-overlap tests pass.

- [ ] **Step 7: Commit the assignment and assembly slice**

```powershell
rtk git -c safe.directory=E:/work/ProLoc-IHCS add -- dataset/download/supplemental_quality_manifests.py dataset/download/test_supplemental_quality_manifests.py
rtk git -c safe.directory=E:/work/ProLoc-IHCS diff --cached --check
rtk git -c safe.directory=E:/work/ProLoc-IHCS commit -m "feat: assign supplemental proteins deterministically"
```

---

### Task 3: CLI sequence policy and atomic six-manifest publication

**Files:**
- Modify: `dataset/download/supplemental_quality_manifests.py`
- Modify: `dataset/download/test_supplemental_quality_manifests.py`

**Interfaces:**
- Consumes: the pinned source functions, `fetch_reviewed_sequences(protein_ids, cache_path)`, and an output directory.
- Produces: `publish_quality_bundle(outputs, output_dir, replace=os.replace) -> None`, `generate_quality_manifests(output_dir, cache_dir, seed, source_urls, source_md5, sequence_resolver) -> tuple[dict[str, pd.DataFrame], dict, pd.DataFrame]`, and `main(argv=None, *, source_urls=None, source_md5=None, sequence_resolver=None) -> int`.

- [ ] **Step 1: Write the comprehensive CLI fixture test and verify RED**

The fixture must include official HQ rows in non-source order, supplemental HQ/MQ/LQ rows, known train/test proteins, ten unknown proteins, a blank Protein Id, a supplemental unresolved protein, and two different proteins sharing the same derived URL. Invoke `main` twice with the same seed, fixture MD5 catalog, and injected resolver, using two output directories. Assert:

- both exit codes are zero;
- HQ row IDs and order exactly match the official fixture;
- the exact official source rows never appear in MQ/LQ;
- the supplemental HQ row appears in MQ;
- known proteins inherit their official side;
- exactly one of ten unknown proteins enters test;
- every normalized protein appears on only one side across all six files;
- every intended shared-URL record remains;
- the unresolved and blank rows appear in `manifest_failures.csv` and not in a manifest;
- the six formal CSV byte sequences and the JSON report are identical between runs.

Run:

```powershell
rtk python -m unittest dataset.download.test_supplemental_quality_manifests.SupplementalManifestCliTest.test_cli_publishes_deterministic_six_manifest_fixture -v
```

Expected: failure because the CLI orchestration does not exist.

- [ ] **Step 2: Implement supplemental row failures after split assignment**

Resolve the union of normalized official and prepared supplemental IDs exactly once. Use `collect_sequence_failures` for official IDs; any official unresolved result is fatal and must leave existing formal manifests untouched. For each unresolved supplemental row, append a failure with `stage=sequence`, its final MQ/LQ tier and already-computed split, then remove the row. Validate `Antibody Id` and `URL` with `build_image_fields`; report invalid supplemental rows as `stage=image_fields` and remove only those rows. Do not recompute `split_mapping` after either removal.

- [ ] **Step 3: Implement the generator and deterministic audit artifacts**

The generator order is fixed:

```python
prepare_verified_sources(cache_dir, source_urls, source_md5)
source = pd.read_csv(cache_dir / "normalLabeled.csv")
official_train = pd.read_csv(cache_dir / "data_train.csv")
official_test = pd.read_csv(cache_dir / "data_test.csv")
official_row_ids = validate_official_rows(source, official_train, official_test)
supplemental, failures, preparation_stats = prepare_supplemental_rows(source, official_row_ids)
split_mapping, split_stats = assign_protein_splits(supplemental, official_train, official_test, seed)
```

After sequence/image validation, assemble all six frames, call `assert_protein_disjoint`, and create `manifest_failures.csv` using `MANIFEST_FAILURE_COLUMNS`. Write `manifest_generation_report.json` with sorted JSON keys and stable values: status, published, record ID, seed, source/preparation/split statistics, output row counts, failure row count, and `protein_id_overlap: 0`.

- [ ] **Step 4: Add and implement atomic publisher rollback tests**

Seed all six destination files with sentinel bytes. Inject a `replace` function that fails while moving the third staged manifest. Assert the exception is raised, every sentinel destination is restored, and no staging or backup directory remains. `publish_quality_bundle` must validate exact output keys, exact `OUTPUT_COLUMNS`, and global Protein Id disjointness before it creates the staging directory; then stage six CSVs, back up existing destinations, replace all destinations, restore all originals on failure, and remove staging/backups in `finally`.

- [ ] **Step 5: Implement CLI success and fatal reporting**

Expose only `--output-dir`, optional `--cache-dir`, and `--seed`. On success, atomically publish the six manifests, then atomically write the deterministic failure CSV and report JSON and print the report. On official/source/global-invariant failure, write `manifest_generation_report.json` with `status=error` and `published=false`, preserve any previous six manifests, print the report to stderr, and return `1`.

- [ ] **Step 6: Run the complete new module tests**

```powershell
rtk python -m unittest dataset.download.test_supplemental_quality_manifests -v
```

Expected: all tests pass with `OK`, without network access.

- [ ] **Step 7: Compile and inspect the CLI**

```powershell
rtk python -m py_compile dataset/download/supplemental_quality_manifests.py dataset/download/test_supplemental_quality_manifests.py
rtk python dataset/download/supplemental_quality_manifests.py --help
```

Expected: compilation exits zero; help lists `--output-dir`, `--cache-dir`, and `--seed` and no historical-HQ or image-download arguments.

- [ ] **Step 8: Commit the CLI slice**

```powershell
rtk git -c safe.directory=E:/work/ProLoc-IHCS add -- dataset/download/supplemental_quality_manifests.py dataset/download/test_supplemental_quality_manifests.py
rtk git -c safe.directory=E:/work/ProLoc-IHCS diff --cached --check
rtk git -c safe.directory=E:/work/ProLoc-IHCS commit -m "feat: publish supplemental quality manifests"
```

---

### Task 4: Regression verification, fixed-base review, and ticket closure

**Files:**
- Modify: `.scratch/official-hq-rgb-dataset/issues/03-publish-supplemental-mq-lq-without-protein-leakage.md`

**Interfaces:**
- Consumes: completed implementation commits and fixed review base `87aac9f3a3fb85c9ee645a724236b86c6e4322ef`.
- Produces: passing regression evidence, resolved blocking review findings, and ticket 03 marked done.

- [ ] **Step 1: Run targeted dependency and feature tests**

```powershell
rtk python -m unittest dataset.download.test_official_hq_manifests dataset.download.test_supplemental_quality_manifests -v
rtk python -m unittest dataset.download.test_filter_quality_urls.FilterQualityUrlsTest.test_pinned_source_catalog_uses_the_approved_record_and_md5_values dataset.download.test_filter_quality_urls.FilterQualityUrlsTest.test_source_cli_reuses_valid_cache_without_http_request dataset.download.test_filter_quality_urls.FilterQualityUrlsTest.test_source_cli_rejects_official_row_identity_anomalies -v
```

Expected: both invocations end with `OK`.

- [ ] **Step 2: Run repository-level applicable checks**

```powershell
rtk python -m unittest discover -v
rtk python -m compileall -q dataset/download
rtk git -c safe.directory=E:/work/ProLoc-IHCS diff --check 87aac9f3a3fb85c9ee645a724236b86c6e4322ef -- dataset/download/supplemental_quality_manifests.py dataset/download/test_supplemental_quality_manifests.py
```

Expected: tests and compilation exit zero and the diff check prints no errors.

- [ ] **Step 3: Review from the fixed base and repair blocking findings**

Invoke the repository `code-review` skill with fixed point `87aac9f3a3fb85c9ee645a724236b86c6e4322ef`, the ticket 03 checklist, and the approved Official-HQ-aligned RGB specification. Apply only actionable ticket-scoped fixes, rerun the commands from Steps 1-2, and commit any review fix as `fix: harden supplemental manifest publication`.

- [ ] **Step 4: Mark only ticket 03 done**

Change `**Status:** ready-for-agent` to `**Status:** done` and replace the ten ticket 03 checklist markers with `[x]`. Do not modify another issue file.

- [ ] **Step 5: Commit ticket closure and inspect final scope**

```powershell
rtk git -c safe.directory=E:/work/ProLoc-IHCS add -- .scratch/official-hq-rgb-dataset/issues/03-publish-supplemental-mq-lq-without-protein-leakage.md
rtk git -c safe.directory=E:/work/ProLoc-IHCS diff --cached --check
rtk git -c safe.directory=E:/work/ProLoc-IHCS commit -m "docs: mark supplemental manifest ticket done"
rtk git -c safe.directory=E:/work/ProLoc-IHCS status --short
rtk git -c safe.directory=E:/work/ProLoc-IHCS log -5 --oneline
```

Expected: the status still shows every baseline dirty path unchanged, no ticket-owned path is left uncommitted, and the latest commits contain only the plan, new module/tests, review fixes if needed, and ticket 03 status.
