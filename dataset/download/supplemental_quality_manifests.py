import heapq
import random

import pandas as pd

if __package__:
    from .official_hq_manifests import (
        DOWNLOAD_LABEL_COLUMNS,
        OUTPUT_COLUMNS,
        SOURCE_COLUMNS,
        SOURCE_ROW_ID,
        _clean_text,
        _source_row_identity,
        assemble_official_hq,
        build_image_fields,
    )
else:
    from official_hq_manifests import (
        DOWNLOAD_LABEL_COLUMNS,
        OUTPUT_COLUMNS,
        SOURCE_COLUMNS,
        SOURCE_ROW_ID,
        _clean_text,
        _source_row_identity,
        assemble_official_hq,
        build_image_fields,
    )


LABEL_COLUMNS = list(DOWNLOAD_LABEL_COLUMNS)
MANIFEST_FAILURE_COLUMNS = [
    "stage",
    "tier",
    "split",
    "source_line",
    "source_row",
    "Protein Id",
    "URL",
    "reason",
]
REQUIRED_OUTPUTS = {
    "HQ_train",
    "HQ_test",
    "MQ_train",
    "MQ_test",
    "LQ_train",
    "LQ_test",
}


def _require_columns(frame, columns, source_name):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source_name} missing required columns: {', '.join(missing)}"
        )


def split_quality_values(value):
    if not isinstance(value, str):
        raise ValueError("quality values must be strings")
    return [part.strip().lower() for part in value.split(";")]


def classify_quality(intensity, quantity):
    intensities = split_quality_values(intensity)
    quantities = split_quality_values(quantity)
    if len(intensities) != len(quantities):
        raise ValueError(
            "intensity and quantity must have the same number of values"
        )

    quality = None
    for pair in zip(intensities, quantities):
        if pair == ("strong", ">75%"):
            return "HQ"
        if pair in (("moderate", ">75%"), ("strong", "75%-25%")):
            quality = "MQ"
        elif quality is None and pair in (
            ("weak", ">75%"),
            ("moderate", "75%-25%"),
            ("weak", "75%-25%"),
        ):
            quality = "LQ"
    return quality


def _failure_record(row, stage, reason, *, tier="", split=""):
    return {
        "stage": stage,
        "tier": tier,
        "split": split,
        "source_line": int(row["__source_line"]),
        "source_row": _source_row_identity(row.get(SOURCE_ROW_ID)),
        "Protein Id": _clean_text(row.get("Protein Id")),
        "URL": _clean_text(row.get("URL")),
        "reason": reason,
    }


def prepare_supplemental_rows(source, official_row_ids):
    required = [
        SOURCE_ROW_ID,
        "Protein Id",
        "IF Verification",
        "locations",
        "Intensity Level",
        "Quantity",
        *LABEL_COLUMNS,
    ]
    _require_columns(source, required, "normalLabeled.csv")

    working = source.copy()
    working["__source_line"] = range(2, len(working) + 2)
    normalized_official_ids = {
        _source_row_identity(source_row) for source_row in official_row_ids
    }
    source_row_ids = working[SOURCE_ROW_ID].map(_source_row_identity)
    supplemental = working.loc[
        ~source_row_ids.isin(normalized_official_ids)
    ].copy()

    labels = supplemental[LABEL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    eligible = supplemental.loc[
        supplemental["IF Verification"].eq("enhanced")
        & supplemental["locations"].notna()
        & supplemental["locations"].astype(str).str.strip().ne("")
        & labels.notna().all(axis=1)
        & labels.gt(0).any(axis=1)
    ].copy()

    failures = []
    keep_positions = []
    qualities = []
    invalid_quality_rows = 0
    for position, (_, row) in enumerate(eligible.iterrows()):
        try:
            quality = classify_quality(
                row["Intensity Level"], row["Quantity"]
            )
        except ValueError as error:
            invalid_quality_rows += 1
            failures.append(_failure_record(row, "quality", str(error)))
            continue
        if quality is not None:
            keep_positions.append(position)
            qualities.append(quality)

    classified = eligible.iloc[keep_positions].copy()
    classified["Quality"] = pd.Series(
        qualities, index=classified.index, dtype=object
    )
    classified["__protein_id"] = classified["Protein Id"].map(_clean_text)

    demoted = int(classified["Quality"].eq("HQ").sum())
    classified.loc[classified["Quality"].eq("HQ"), "Quality"] = "MQ"

    blank_mask = classified["__protein_id"].eq("")
    blank_rows = classified.loc[blank_mask]
    for _, row in blank_rows.iterrows():
        failures.append(
            _failure_record(
                row,
                "protein_id",
                "Protein Id is blank after trimming leading/trailing whitespace",
                tier=row["Quality"],
            )
        )
    classified = classified.loc[~blank_mask].copy()

    failures.sort(key=lambda failure: failure["source_line"])
    stats = {
        "source_rows": len(source),
        "supplemental_source_rows": len(supplemental),
        "eligible_rows": len(eligible),
        "classified_rows": len(classified),
        "invalid_quality_rows": invalid_quality_rows,
        "blank_protein_rows": len(blank_rows),
        "supplemental_hq_demoted": demoted,
    }
    return classified, failures, stats


def _normalized_protein_set(frame, source_name):
    _require_columns(frame, ["Protein Id"], source_name)
    return {
        protein_id
        for protein_id in frame["Protein Id"].map(_clean_text)
        if protein_id
    }


def _select_test_proteins(protein_labels, unknown, seed):
    if not unknown:
        return set()

    test_count = min(len(unknown), max(1, round(len(unknown) * 0.1)))
    target = protein_labels.loc[unknown].sum() * 0.1
    totals = pd.Series(0.0, index=protein_labels.columns)
    rng = random.Random(seed)
    tie_values = {protein_id: rng.random() for protein_id in unknown}
    buckets = {}
    vectors = {}
    for protein_id in unknown:
        signature = tuple(
            bool(value) for value in protein_labels.loc[protein_id]
        )
        vectors.setdefault(
            signature,
            pd.Series(signature, index=protein_labels.columns, dtype=float),
        )
        heapq.heappush(
            buckets.setdefault(signature, []),
            (tie_values[protein_id], protein_id),
        )

    selected = set()
    for _ in range(test_count):
        _, _, protein_id, signature = min(
            (
                float(((totals + vectors[signature] - target) ** 2).sum()),
                bucket[0][0],
                bucket[0][1],
                signature,
            )
            for signature, bucket in buckets.items()
            if bucket
        )
        heapq.heappop(buckets[signature])
        selected.add(protein_id)
        totals += vectors[signature]
    return selected


def assign_protein_splits(supplemental, official_train, official_test, seed):
    _require_columns(
        supplemental,
        ["__protein_id", *LABEL_COLUMNS],
        "supplemental rows",
    )
    official_train_ids = _normalized_protein_set(
        official_train, "data_train.csv"
    )
    official_test_ids = _normalized_protein_set(
        official_test, "data_test.csv"
    )
    overlap = sorted(official_train_ids & official_test_ids)
    if overlap:
        raise ValueError(
            f"official train and test Protein Id overlap: {overlap}"
        )

    supplemental_ids = set(supplemental["__protein_id"])
    known_train = supplemental_ids & official_train_ids
    known_test = supplemental_ids & official_test_ids
    unknown = sorted(
        supplemental_ids - official_train_ids - official_test_ids
    )

    mapping = {
        **{protein_id: "train" for protein_id in official_train_ids},
        **{protein_id: "test" for protein_id in official_test_ids},
    }
    if unknown:
        grouped = supplemental.loc[
            supplemental["__protein_id"].isin(unknown),
            ["__protein_id", *LABEL_COLUMNS],
        ].copy()
        grouped[LABEL_COLUMNS] = grouped[LABEL_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
        protein_labels = (
            grouped.groupby("__protein_id", sort=True)[LABEL_COLUMNS]
            .max()
            .gt(0)
        )
        selected_test = _select_test_proteins(
            protein_labels, unknown, seed
        )
        mapping.update(
            {
                protein_id: (
                    "test" if protein_id in selected_test else "train"
                )
                for protein_id in unknown
            }
        )
    else:
        selected_test = set()

    stats = {
        "known_train_proteins": len(known_train),
        "known_test_proteins": len(known_test),
        "unknown_proteins": len(unknown),
        "unknown_test_proteins": len(selected_test),
    }
    return mapping, stats


def _manifest_rows(frame, sequences, source_name):
    _require_columns(frame, SOURCE_COLUMNS, source_name)
    result = frame.copy()
    protein_ids = result["Protein Id"].map(_clean_text)
    missing_sequences = sorted(
        {
            protein_id or "<blank>"
            for protein_id in protein_ids
            if not protein_id or not _clean_text(sequences.get(protein_id))
        }
    )
    if missing_sequences:
        raise ValueError(
            "missing reviewed sequences for: "
            + ", ".join(missing_sequences)
        )

    result["Sequence"] = protein_ids.map(sequences)
    image_fields = [
        build_image_fields(antibody_id, relative_url)
        for antibody_id, relative_url in result[
            ["Antibody Id", "URL"]
        ].itertuples(index=False, name=None)
    ]
    result[["NonZeroDigits", "Image Name", "Modified URL"]] = pd.DataFrame(
        image_fields,
        columns=["NonZeroDigits", "Image Name", "Modified URL"],
        index=result.index,
    )
    return result.loc[:, OUTPUT_COLUMNS]


def _manifest_proteins(frame, source_name):
    _require_columns(frame, ["Protein Id"], source_name)
    return {
        protein_id
        for protein_id in frame["Protein Id"].map(_clean_text)
        if protein_id
    }


def assert_protein_disjoint(outputs):
    if set(outputs) != REQUIRED_OUTPUTS:
        raise ValueError(f"unexpected output keys: {sorted(outputs)}")
    train_proteins = set().union(
        *(
            _manifest_proteins(frame, name)
            for name, frame in outputs.items()
            if name.endswith("_train")
        )
    )
    test_proteins = set().union(
        *(
            _manifest_proteins(frame, name)
            for name, frame in outputs.items()
            if name.endswith("_test")
        )
    )
    overlap = sorted(train_proteins & test_proteins)
    if overlap:
        raise AssertionError(
            f"Protein Id overlap between train and test: {overlap}"
        )


def assemble_quality_outputs(
    official_train,
    official_test,
    supplemental,
    split_mapping,
    sequences,
):
    _require_columns(
        supplemental,
        ["Quality", "__protein_id"],
        "supplemental rows",
    )
    missing_splits = sorted(
        set(supplemental["__protein_id"]) - set(split_mapping)
    )
    if missing_splits:
        raise ValueError(
            f"missing split assignments for: {', '.join(missing_splits)}"
        )
    row_splits = supplemental["__protein_id"].map(split_mapping)
    invalid_splits = sorted(set(row_splits) - {"train", "test"})
    if invalid_splits:
        raise ValueError(f"invalid split assignments: {invalid_splits}")

    outputs = assemble_official_hq(
        official_train, official_test, sequences
    )
    for tier in ("MQ", "LQ"):
        for split in ("train", "test"):
            subset = supplemental.loc[
                supplemental["Quality"].eq(tier) & row_splits.eq(split)
            ]
            outputs[f"{tier}_{split}"] = _manifest_rows(
                subset, sequences, f"{tier}_{split} supplemental rows"
            )
    assert_protein_disjoint(outputs)
    return outputs
