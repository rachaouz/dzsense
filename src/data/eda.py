import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)
NAMES = {0: "negative", 1: "neutral", 2: "positive"}

train = pd.read_parquet("data/processed/train.parquet")
train["class"] = train["label"].map(NAMES)
train["n_words"] = train["text_norm"].str.split().str.len()

# 1. classes
fig, ax = plt.subplots(figsize=(6, 4))
train["class"].value_counts().plot.bar(ax=ax)
ax.set_title("Distribution des classes (train)")
fig.tight_layout(); fig.savefig(FIG / "class_distribution.png", dpi=150); plt.close(fig)

# 2. longueur des commentaires par classe
fig, ax = plt.subplots(figsize=(8, 4))
for c, g in train.groupby("class"):
    g["n_words"].clip(upper=100).plot.hist(bins=50, alpha=0.5, label=c, ax=ax)
ax.set_xlabel("Nombre de mots (tronqué à 100)"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "length_by_class.png", dpi=150); plt.close(fig)

# 3. mélange de scripts
def share(text, pattern):
    letters = re.findall(r"[^\W\d_]", text)
    return len(re.findall(pattern, text)) / len(letters) if letters else 0.0

train["arabic_share"] = train["text_norm"].map(lambda t: share(t, r"[\u0600-\u06FF]"))
train["latin_share"] = train["text_norm"].map(lambda t: share(t, r"[a-z]"))
summary = pd.DataFrame({
    "mostly_arabic_script_%": [(train["arabic_share"] > 0.8).mean() * 100],
    "mostly_latin_script_%": [(train["latin_share"] > 0.8).mean() * 100],
    "mixed_%": [(train[["arabic_share", "latin_share"]].max(axis=1) <= 0.8).mean() * 100],
}).round(1)
print(summary.to_string(index=False))
summary.to_csv("reports/script_mix.csv", index=False)

# 4. mots les plus fréquents par classe
rows = []
for c, g in train.groupby("class"):
    cnt = Counter(w for t in g["text_norm"] for w in t.split())
    rows += [{"class": c, "word": w, "count": n} for w, n in cnt.most_common(30)]
pd.DataFrame(rows).to_csv("reports/top_words_by_class.csv", index=False, encoding="utf-8-sig")

print(train.groupby("class")["n_words"].describe().round(1))