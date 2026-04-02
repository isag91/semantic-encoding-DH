
#!/usr/bin/env python3
"""
Compute one-hot and semantic encodings for the digital literature dataset,
then output:

1. Pairwise cosine distances between genre centroids
2. Normalized centrality of database centroids:
   - RGD = ratio to global dispersion
   - RO  = relative offset

This script is tailored to the attached dataset format:
- metadata in wide binary format (e.g. "Genre : poetry", "Format : image", ...)
- semantic descriptions in a JSON mapping from "feature_value" keys to text

Example
-------
python dh_semantic_pipeline_adjusted.py \
    --csv /path/to/data_DL_.csv \
    --descriptions /path/to/descriptions_DL.JSON \
    --outdir /path/to/outputs \
    --drop-features Technique

Optional:
- add --no-technique as a shortcut to drop the Technique feature
- change --model if you want another Hugging Face encoder
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------
# Configuration tailored to the attached files
# ---------------------------------------------------------------------

NON_METADATA_COLUMNS = {
    "Title", "Author(s)", "Country", "Year", "DB", "URL", "Women", "Men", "Both"
}


GENRE_LABELS_DEFAULT = [
    "poetry",
    "narrative",
    "poetry and narrative",
]


NO_INFORMATION_KEYS = {
    "Format": "format_no_information",
    "Genre": "genre_no_information",
    "Access hardware": "access_hardware_no_information",
    "Publication type": "publication_type_no_information",
    "Program": "program_no_information",
    "Technical requirements": "technical_requirements_no_information",
    "Reading process": "reading_process_no_information",
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def metadata_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c in NON_METADATA_COLUMNS:
            continue
        if " : " in c:
            cols.append(c)
    return cols


def split_feature_value(col: str) -> Tuple[str, str]:
    feature, value = col.split(" : ", 1)
    return normalize_spaces(feature), normalize_spaces(value)


def to_description_key(feature: str, value: str) -> str:
    key = f"{feature.lower()}_{value.lower()}"
    key = key.replace(" ", "_")
    return key


def get_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}

    for col in metadata_columns(df):
        feature, _ = split_feature_value(col)
        if feature not in groups:
            groups[feature] = []
        groups[feature].append(col)

    return groups


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    sim = float(np.dot(a, b) / (na * nb))
    sim = max(min(sim, 1.0), -1.0)
    return 1.0 - sim


# ---------------------------------------------------------------------
# Transformer encoding: token-activation weighted pooling
# ---------------------------------------------------------------------

class AttentionWeightedTextEncoder:
    """
    Implements the parameter-free token weighting described in the spirit of ARISE:
    1) obtain token embeddings from the last hidden state
    2) compute token scores as mean activation across hidden dimensions
    3) softmax over scores
    4) weighted sum of token embeddings
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    @torch.inference_mode()
    def encode_text(self, text: str, max_length: int = 256) -> np.ndarray:
        batch = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        hidden = outputs.last_hidden_state[0]               # [seq_len, hidden_dim]
        attention_mask = batch["attention_mask"][0].bool()  # [seq_len]

        hidden = hidden[attention_mask]
        if hidden.shape[0] == 0:
            # extremely defensive fallback
            hidden = outputs.last_hidden_state[0]

        scores = hidden.mean(dim=1)            # [seq_len]
        weights = torch.softmax(scores, dim=0) # [seq_len]
        emb = (weights.unsqueeze(1) * hidden).sum(dim=0)
        return emb.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------
# Build encodings
# ---------------------------------------------------------------------


def precompute_value_embeddings(
    feature_groups: Dict[str, List[str]],
    descriptions: Dict[str, str],
    encoder: AttentionWeightedTextEncoder,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Precompute one semantic embedding per metadata value column.
    Also precompute feature-level no_information embeddings where available.
    """
    embeddings: Dict[str, np.ndarray] = {}
    keys_to_encode: Dict[str, str] = {}

    for feature, cols in feature_groups.items():
        for col in cols:
            f, v = split_feature_value(col)
            raw_key = to_description_key(f, v)
            if raw_key not in descriptions:
                raise KeyError(
                    f"Missing description for column '{col}' "
                    f"(expected key '{raw_key}')."
                )
            keys_to_encode[col] = raw_key

        no_info_key = NO_INFORMATION_KEYS.get(feature)
        if no_info_key and no_info_key in descriptions:
            keys_to_encode[f"__NO_INFORMATION__::{feature}"] = no_info_key

    total = len(keys_to_encode)
    for idx, (name, desc_key) in enumerate(keys_to_encode.items(), start=1):
        if verbose:
            print(f"[semantic] encoding {idx}/{total}: {name} -> {desc_key}")
        embeddings[name] = encoder.encode_text(descriptions[desc_key])

    return embeddings


def build_onehot_encoding(
    df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    add_no_information: bool = True,
) -> pd.DataFrame:
    """
    Keep the existing binary columns and optionally append one synthetic
    no_information column per feature:
    value is 1 if the row has no active value within that feature.
    """
    parts = []
    for feature, cols in feature_groups.items():
        block = df[cols].fillna(0).astype(np.float32).copy()

        if add_no_information:
            no_info_col = f"{feature} : no_information"
            active = block.sum(axis=1)
            block[no_info_col] = (active == 0).astype(np.float32)

        parts.append(block)

    encoded = pd.concat(parts, axis=1)
    encoded.index = df.index
    return encoded


def build_semantic_encoding(
    df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    value_embeddings: Dict[str, np.ndarray],
    add_no_information: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build one semantic block per feature by averaging the embeddings of all
    active values in that feature. If no value is active and a no_information
    embedding exists, use it.
    """
    row_vectors = []
    feature_labels = []
    dim_per_feature = None

    for feature, cols in feature_groups.items():
        sample_key = cols[0]
        feature_dim = int(value_embeddings[sample_key].shape[0])
        dim_per_feature = feature_dim if dim_per_feature is None else dim_per_feature
        if dim_per_feature != feature_dim:
            raise ValueError("All semantic embeddings must have same dimensionality.")

        feature_labels.extend([f"{feature}__dim_{i}" for i in range(feature_dim)])

    for _, row in df.iterrows():
        blocks = []
        for feature, cols in feature_groups.items():
            active_cols = []
            for col in cols:
                val = row[col]
                if pd.notna(val) and float(val) > 0:
                    active_cols.append(col)

            if active_cols:
                vecs = np.stack([value_embeddings[c] for c in active_cols], axis=0)
                block = vecs.mean(axis=0)
            else:
                no_info_name = f"__NO_INFORMATION__::{feature}"
                if add_no_information and no_info_name in value_embeddings:
                    block = value_embeddings[no_info_name]
                else:
                    block = np.zeros_like(value_embeddings[cols[0]], dtype=np.float32)

            blocks.append(block.astype(np.float32))

        row_vectors.append(np.concatenate(blocks, axis=0))

    mat = np.vstack(row_vectors)
    sem_df = pd.DataFrame(mat, index=df.index, columns=feature_labels)
    return sem_df, feature_labels


# ---------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------

def compute_genre_centroid_pairwise_distances(
    df: pd.DataFrame,
    encoded: pd.DataFrame,
    genre_labels: List[str],
) -> pd.DataFrame:
    rows = []
    centroids: Dict[str, np.ndarray] = {}

    for label in genre_labels:
        genre_col = f"Genre : {label}"
        if genre_col not in df.columns:
            raise KeyError(f"Genre column not found: {genre_col}")
        mask = df[genre_col].fillna(0).astype(float) > 0
        if mask.sum() == 0:
            raise ValueError(f"No rows found for genre label '{label}'")
        centroids[label] = encoded.loc[mask].mean(axis=0).to_numpy(dtype=np.float64)

    for a, b in combinations(genre_labels, 2):
        rows.append({
            "label_a": a,
            "label_b": b,
            "cosine_distance": cosine_distance(centroids[a], centroids[b]),
            "n_a": int((df[f"Genre : {a}"].fillna(0).astype(float) > 0).sum()),
            "n_b": int((df[f"Genre : {b}"].fillna(0).astype(float) > 0).sum()),
        })

    return pd.DataFrame(rows)


def compute_database_normalized_centrality(
    df: pd.DataFrame,
    encoded: pd.DataFrame,
    database_col: str = "DB",
) -> pd.DataFrame:
    if database_col not in df.columns:
        raise KeyError(f"Database column not found: {database_col}")

    X = encoded.to_numpy(dtype=np.float64)
    global_centroid = X.mean(axis=0)

    global_distances = np.array([cosine_distance(x, global_centroid) for x in X], dtype=np.float64)
    mean_global_dispersion = float(global_distances.mean()) if len(global_distances) else 0.0

    rows = []
    for db_name, idx in df.groupby(database_col).groups.items():
        idx = list(idx)
        X_db = X[idx]
        db_centroid = X_db.mean(axis=0)

        d_db_to_global = cosine_distance(db_centroid, global_centroid)
        db_internal = np.array([cosine_distance(x, db_centroid) for x in X_db], dtype=np.float64)
        mean_db_internal = float(db_internal.mean()) if len(db_internal) else 0.0

        rgd = d_db_to_global / mean_global_dispersion if mean_global_dispersion > 0 else np.nan
        ro = d_db_to_global / mean_db_internal if mean_db_internal > 0 else np.nan

        rows.append({
            "database": db_name,
            "n_works": len(idx),
            "distance_to_global_centroid": d_db_to_global,
            "mean_global_dispersion": mean_global_dispersion,
            "mean_internal_dispersion": mean_db_internal,
            "RGD": rgd,
            "RO": ro,
        })

    return pd.DataFrame(rows).sort_values("database").reset_index(drop=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to data_DL_.csv")
    parser.add_argument("--descriptions", required=True, help="Path to descriptions_DL.JSON")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Hugging Face model for semantic encoding",
    )
 
    parser.add_argument(
        "--database-col",
        default="DB",
        help="Column containing the database identifier",
    )
    parser.add_argument(
        "--genre-labels",
        nargs="*",
        default=GENRE_LABELS_DEFAULT,
        help='Genre labels to compare, e.g. "poetry" "narrative" "poetry and narrative"',
    )
    parser.add_argument(
        "--no-no-information",
        action="store_true",
        help="Do not add synthetic no_information dimensions/embeddings per feature",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)

    df = pd.read_csv(args.csv)
    with open(args.descriptions, "r", encoding="utf-8") as f:
        descriptions: Dict[str, str] = json.load(f)

    feature_groups = get_feature_groups(df)
    if not feature_groups:
        raise ValueError("No metadata feature groups found in the dataset.")

    print("[info] selected feature groups:")
    for feature, cols in feature_groups.items():
        print(f"  - {feature}: {len(cols)} columns")

    add_no_information = not args.no_no_information

    # One-hot
    onehot_df = build_onehot_encoding(
        df=df,
        feature_groups=feature_groups,
        add_no_information=add_no_information,
    )

    # Semantic
    encoder = AttentionWeightedTextEncoder(model_name=args.model)
    value_embeddings = precompute_value_embeddings(
        feature_groups=feature_groups,
        descriptions=descriptions,
        encoder=encoder,
        verbose=True,
    )
    semantic_df, _ = build_semantic_encoding(
        df=df,
        feature_groups=feature_groups,
        value_embeddings=value_embeddings,
        add_no_information=add_no_information,
    )

    # Analyses
    genre_oh = compute_genre_centroid_pairwise_distances(df, onehot_df, args.genre_labels)
    genre_sem = compute_genre_centroid_pairwise_distances(df, semantic_df, args.genre_labels)
    db_oh = compute_database_normalized_centrality(df, onehot_df, args.database_col)
    db_sem = compute_database_normalized_centrality(df, semantic_df, args.database_col)

    genre_compare = genre_oh.merge(
        genre_sem,
        on=["label_a", "label_b", "n_a", "n_b"],
        suffixes=("_onehot", "_semantic"),
    )

    db_compare = db_oh.merge(
        db_sem,
        on=["database", "n_works"],
        suffixes=("_onehot", "_semantic"),
    )

    # Save outputs
    onehot_path = os.path.join(args.outdir, "onehot_encoding.csv")
    semantic_path = os.path.join(args.outdir, "semantic_encoding.csv")
    genre_path = os.path.join(args.outdir, "genre_centroid_pairwise_cosine_distances.csv")
    db_path = os.path.join(args.outdir, "database_centroid_normalized_centrality.csv")
    info_path = os.path.join(args.outdir, "run_info.json")

    onehot_df.to_csv(onehot_path, index=False)
    semantic_df.to_csv(semantic_path, index=False)
    genre_compare.to_csv(genre_path, index=False)
    db_compare.to_csv(db_path, index=False)

    run_info = {
        "input_csv": args.csv,
        "descriptions": args.descriptions,
        "semantic_model": args.model,
        "add_no_information": add_no_information,
        "feature_groups": list(feature_groups.keys()),
        "genre_labels": args.genre_labels,
        "database_col": args.database_col,
        "n_rows": int(len(df)),
        "n_onehot_dims": int(onehot_df.shape[1]),
        "n_semantic_dims": int(semantic_df.shape[1]),
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print("\n[done] files written:")
    print(f"  - {onehot_path}")
    print(f"  - {semantic_path}")
    print(f"  - {genre_path}")
    print(f"  - {db_path}")
    print(f"  - {info_path}")


if __name__ == "__main__":
    main()
