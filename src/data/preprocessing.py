"""Chargement, nettoyage et découpage du dataset de sentiment en dialecte algérien."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

HF_ID = "Abdou/dz-sentiment-yt-comments"
RAW_PATH = Path("data/raw/raw.parquet")
OUT_DIR = Path("data/processed")
REPORT = Path("reports/data_report.json")
SEED = 42
LABELS = {0: "negative", 1: "neutral", 2: "positive"}

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")
ALEF = re.compile(r"[إأآ]")
ELONGATION = re.compile(r"(.)\1{2,}")


def normalize(text: str) -> str:
    """Normalisation légère pour les modèles classiques (TF-IDF).
    On garde le texte brut séparément pour le Transformer.
    On ne touche pas ى/ي ni ة/ه : en dialecte ils portent du sens."""
    t = unicodedata.normalize("NFKC", str(text))
    t = URL.sub(" ", t)
    t = MENTION.sub(" ", t)
    t = DIACRITICS.sub("", t).replace("\u0640", "")   # diacritiques + tatweel
    t = ALEF.sub("ا", t)
    t = ELONGATION.sub(r"\1\1", t)                    # "جمييييل" -> "جمييل"
    t = t.lower()
    return re.sub(r"\s+", " ", t).strip()


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        from datasets import concatenate_datasets, load_dataset
        ds = load_dataset(HF_ID)
        df = concatenate_datasets([ds[k] for k in ds]).to_pandas()
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(RAW_PATH)
    return pd.read_parquet(RAW_PATH)


def find_columns(df: pd.DataFrame) -> tuple[str, str]:
    print("Colonnes trouvées :", list(df.columns))
    if "label" not in df.columns:
        raise ValueError("Colonne 'label' absente : dis-moi les colonnes affichées.")
    text_col = "text" if "text" in df.columns else next(
        c for c in df.columns if c != "label" and df[c].dtype == object)
    return text_col, "label"


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    text_col, label_col = find_columns(df)
    rep = {"rows_raw": len(df), "text_column": text_col}
    df = df[[text_col, label_col]].rename(columns={text_col: "text_raw", label_col: "label"})

    bad = set(df["label"].unique()) - set(LABELS)
    if bad:
        raise ValueError(f"Labels inattendus : {bad}")

    rep["missing_text"] = int(df["text_raw"].isna().sum())
    df = df.dropna(subset=["text_raw"])
    df["text_norm"] = df["text_raw"].map(normalize)

    empty = df["text_norm"].str.len() == 0
    rep["empty_after_normalization"] = int(empty.sum())
    df = df.loc[~empty]

    # même texte, labels différents -> ambigu, on retire
    n_labels = df.groupby("text_norm")["label"].transform("nunique")
    rep["conflicting_label_rows_dropped"] = int((n_labels > 1).sum())
    df = df.loc[n_labels == 1]

    n = len(df)
    df = df.drop_duplicates(subset="text_norm")
    rep["duplicates_dropped"] = n - len(df)

    rep["rows_clean"] = len(df)
    rep["label_counts"] = {LABELS[k]: int(v) for k, v in df["label"].value_counts().items()}
    return df.reset_index(drop=True), rep


def split(df: pd.DataFrame):
    train, tmp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=SEED)
    return train, val, test


def main():
    raw = load_raw()
    rep_hash = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
    df, rep = clean(raw)
    train, val, test = split(df)

    # contrôle anti-fuite : aucun texte commun entre partitions
    sets = {k: set(v["text_norm"]) for k, v in [("train", train), ("val", val), ("test", test)]}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) \
        and not (sets["val"] & sets["test"]), "Fuite : texte présent dans 2 partitions"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.reset_index(drop=True).to_parquet(OUT_DIR / f"{name}.parquet")
        rep[f"rows_{name}"] = len(part)
    rep["raw_snapshot_sha256"] = rep_hash
    rep["source"] = HF_ID
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()