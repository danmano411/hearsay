import pandas as pd

from hearsay.data.given import RAW, diffssd_meta, lj_meta
from hearsay.data.splits import HELDOUT_GENERATORS, assign_splits
from hearsay.data.stats import clip_stats


def _toy():
    rows = []
    for n in range(300):
        for gen in ("grad_tts", "diffgan_tts"):  # LJ-voice fakes
            rows.append(("spoof", gen, "ljspeech:LJ", f"diffssd:{n}"))
        for spk in ("librispeech:100", "librispeech:1487"):
            rows.append(("spoof", "xtts_v2", spk, f"diffssd:{n}"))
        rows.append(("bonafide", "bonafide", "ljspeech:LJ", f"diffssd:{n}"))  # real LJ reading the same sentence
        rows.append(("bonafide", "bonafide", None, None))  # external clip without metadata
    df = pd.DataFrame(rows, columns=["label", "generator", "speaker", "text_id"])
    df["source"] = "toy"
    lj = [("lj_real", "bonafide", "bonafide", "ljspeech:LJ", f"ljspeech:LJ001-{i:04d}") for i in range(50)]
    lj += [("sim", "spoof", "sim_vits_ljs", "LJ", f"LJ001-{i:04d}") for i in range(50)]  # other phase's id style
    itw = [("in_the_wild", lab, "itw", f"celeb{i % 5}", f"itw:{i}") for i, lab in enumerate(["bonafide", "spoof"] * 50)]
    df = pd.concat([df, pd.DataFrame(lj + itw, columns=["source", "label", "generator", "speaker", "text_id"])],
                   ignore_index=True)
    df["path"] = [f"p/{i}.wav" for i in range(len(df))]
    return df


def test_splits_leak_free_and_deterministic():
    df = _toy()
    a, b = assign_splits(df), assign_splits(df.sample(frac=1, random_state=0))
    assert a.set_index("path")["split"].equals(b.set_index("path")["split"].loc[a["path"]])  # order-independent
    bucket = a["split"].replace({"excluded": "train"})
    st = a.dropna(subset=["text_id"])
    assert (bucket[st.index].groupby([st["speaker"], st["text_id"]]).nunique() == 1).all()
    assert (bucket[st.index].groupby(st["text_id"]).nunique() == 1).all()  # real LJ + LJ fakes of a sentence together
    lj = a[a["text_id"].str.contains("LJ001", na=False)]
    assert (bucket[lj.index].groupby(lj["text_id"].str[-10:]).nunique() == 1).all()  # real LJ vs sim LJ fake
    itw = a[a.source == "in_the_wild"]
    assert (bucket[itw.index].groupby(itw["speaker"]).nunique() == 1).all()
    assert not (a["generator"].isin(HELDOUT_GENERATORS) & (a["split"] == "train")).any()
    assert set(a["split"]) == {"train", "val", "test_internal", "excluded"}
    assert set(a.loc[a.label == "spoof", "logo_fold"]) == {"grad_tts", "diffgan_tts", "xtts_v2", "sim_vits_ljs", "itw"}
    assert a["logo_fold"].notna().all()
    t = a[a["val_testlike"]]
    assert (t["split"] == "val").all() and abs((t["label"] == "spoof").mean() - 0.3) < 0.05
    # held-out generator preferred: here n_spoof exceeds its val rows, so all of them are taken before any other
    val_heldout = a[(a.split == "val") & (a.generator == "diffgan_tts")]
    assert (t.label == "spoof").sum() > len(val_heldout) and val_heldout.index.isin(t.index).all()


def test_path_metadata():
    m = diffssd_meta(RAW / "diffssd/openvoicev2/speaker_1487/sentence_12_en-au.wav")
    assert (m["speaker"], m["text_id"], m["accent"]) == ("librispeech:1487", "diffssd:12", "en-au")
    assert diffssd_meta(RAW / "diffssd/grad_tts/sentence_4999.wav")["speaker"] == "ljspeech:LJ"
    assert lj_meta(RAW / "lj_real/LJ001-0001.wav")["text_id"] == "ljspeech:LJ001-0001"


def test_clip_stats_silence_and_bandwidth():
    import numpy as np
    sr = 16000
    t = np.arange(sr) / sr
    y = np.concatenate([np.zeros(sr // 2), 0.5 * np.sin(2 * np.pi * 1000 * t), np.zeros(sr // 4)])
    s = clip_stats(y, sr)
    assert abs(s["lead_sil"] - 0.5) < 1e-3 and abs(s["trail_sil"] - 0.25) < 1e-3
    assert 900 < s["rolloff95"] < 1100 and abs(s["peak"] - 0.5) < 1e-6
