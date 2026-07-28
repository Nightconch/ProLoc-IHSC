import pandas as pd

if __package__:
    from .official_hq_manifests import (
        DOWNLOAD_LABEL_COLUMNS,
        SOURCE_ROW_ID,
        _clean_text,
        _source_row_identity,
    )
else:
    from official_hq_manifests import (
        DOWNLOAD_LABEL_COLUMNS,
        SOURCE_ROW_ID,
        _clean_text,
        _source_row_identity,
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
