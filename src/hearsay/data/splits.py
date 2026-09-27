"""Deterministic, leakage-safe split assignment over the union of all source manifests.

Every rule is a pure function of row content (hashes, no RNG), so adding a new source never moves existing rows
between splits, and re-running gives identical output. See docs/02_dataset.md for the rationale.
"""
import hashlib

import numpy as np
import pandas as pd

TRAIN_PCT, VAL_PCT = 80, 10  # remaining 10% -> test_internal
# Generators DiffSSD's own protocol reserves for test only (train_val_test_splits.csv): never seen in training here.
HELDOUT_GENERATORS = ("diffgan_tts", "playht", "unit_speech")
# Multi-speaker corpora whose text_id is a per-file utterance id: group by speaker so a voice (and its recording
# channel) never appears on both sides of a split. In-the-Wild = real + fake clips of the same 54 public figures;
# ASVspoof / LibriSeVoc = LibriSpeech/LibriTTS-derived speakers (LibriSeVoc vocodes each real utterance, same speaker).
SPEAKER_GROUPED_SOURCES = ("in_the_wild", "asvspoof2019_la", "asvspoof5", "librisevoc")
TESTLIKE_SPOOF_FRAC = 0.30  # HGT test is ~70% real / 30% synthetic


def _h(s):
    return int(hashlib.sha1(str(s).encode()).hexdigest()[:12], 16)


def group_key(df):
    """Leakage unit. text_id (spoken sentence) if known: then no sentence -- and hence no (speaker, text_id)
    pair -- ever spans two splits, whatever voice or generator spoke it. Otherwise the speaker, otherwise the file.
    Sources in SPEAKER_GROUPED_SOURCES are grouped by speaker instead (their text_ids are per-file)."""
    text = df["text_id"].astype("string")
    # LJSpeech utterance ids are global: "LJ001-0001" and "ljspeech:LJ001-0001" are the same sentence
    lj = text.str.extract(r"(LJ\d{3}-\d{4})", expand=False)
    text = ("ljspeech:" + lj).fillna(text)
    spk = "spk:" + df["source"].astype("string") + ":" + df["speaker"].astype("string")
    key = text.mask(df["source"].isin(SPEAKER_GROUPED_SOURCES) & spk.notna(), spk)
    key = key.fillna(spk)
    return key.fillna("path:" + df["path"].astype("string")).astype(str)


def assign_splits(df):
    df = df.copy()
    df["group"] = group_key(df)
    bucket = df["group"].map(lambda g: _h(g) % 100)
    df["split"] = np.select([bucket < TRAIN_PCT, bucket < TRAIN_PCT + VAL_PCT], ["train", "val"], "test_internal")
    # held-out generators: rows whose sentence falls in train are parked, not trained on (keeps groups intact)
    df.loc[df["generator"].isin(HELDOUT_GENERATORS) & (df["split"] == "train"), "split"] = "excluded"

    # LOGO: spoof rows fold = their generator; bonafide rows are spread over the folds by group hash, so each fold's
    # eval set = one unseen generator + a disjoint slice of real speech.
    gens = sorted(df.loc[df["label"] == "spoof", "generator"].unique())
    df["logo_fold"] = df["generator"].where(df["label"] == "spoof")
    if gens:
        bona = df["label"] == "bonafide"
        df.loc[bona, "logo_fold"] = df.loc[bona, "group"].map(lambda g: gens[_h("logo" + g) % len(gens)])

    for split in ("val", "test_internal"):
        df[f"{split}_testlike"] = testlike_mask(df, split)
    return df


def testlike_mask(df, split):
    """All bonafide rows of `split` + spoof rows so that spoof = 30%, taking held-out generators first."""
    in_split = df["split"] == split
    n_bona = int((in_split & (df["label"] == "bonafide")).sum())
    n_spoof = round(n_bona * TESTLIKE_SPOOF_FRAC / (1 - TESTLIKE_SPOOF_FRAC))
    spoof = df[in_split & (df["label"] == "spoof")]
    order = pd.DataFrame({"not_heldout": ~spoof["generator"].isin(HELDOUT_GENERATORS),
                          "h": spoof["path"].map(_h)}, index=spoof.index).sort_values(["not_heldout", "h"])
    mask = in_split & (df["label"] == "bonafide")
    mask.loc[order.index[:n_spoof]] = True
    return mask
