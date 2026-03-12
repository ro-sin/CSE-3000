import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    # merge on quasi-identifiers
    merged = pd.merge(
        anon_df,
        aux_df,
        on=["age", "zip3", "gender"],
        how="inner"
    )

    # count how many matches each anonymized record gets
    counts = merged.groupby("anon_id").size().reset_index(name="count")

    # keep only anonymized records that matched exactly once
    unique_ids = counts[counts["count"] == 1]["anon_id"]

    unique_matches = merged[merged["anon_id"].isin(unique_ids)]

    # return only required columns
    return unique_matches[["anon_id", "name"]].rename(columns={"name": "matched_name"})


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    num_matches = len(matches_df)
    total_records = len(anon_df)

    return num_matches / total_records