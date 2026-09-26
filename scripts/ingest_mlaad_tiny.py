"""MLAAD-tiny (Mueller et al., MLAAD v8 subset, CC BY-NC 4.0) -> data/processed/mlaad_tiny/.

English only: ~100 clips from each of ~75 English TTS systems (2023-2025 era: F5, E2, Kokoro, XTTS, Bark,
Chatterbox, VibeVoice, commercial APIs, ...) + the English M-AILABS bona fide audiobook clips.
The full MLAAD repo is gated (click-through), MLAAD-tiny is not.
"""
import re

from huggingface_hub import snapshot_download

from ingest_common import EXTERNAL, out_path, run

REPO = "mueller91/MLAAD-tiny"
SOURCE = "mlaad_tiny"
LICENSE = "CC-BY-NC-4.0 (research only)"


def slug(s):
    return re.sub(r"[^A-Za-z0-9.]+", "_", s).strip("_")


def jobs():
    root = EXTERNAL / SOURCE
    snapshot_download(REPO, repo_type="dataset", local_dir=root, max_workers=4,
                      allow_patterns=["fake/en/**", "original/en/**", "LICENSE", "README.md"])
    for p in sorted((root / "original" / "en").rglob("*.wav")):
        yield dict(src=str(p), out=str(out_path(SOURCE, "bonafide", p.stem)), label="bonafide", source=SOURCE,
                   generator="bonafide", speaker="", text_id=p.stem, codec_orig="wav", license=LICENSE)
    for p in sorted((root / "fake" / "en").rglob("*.wav")):
        gen = "mlaad_" + slug(p.parent.name)
        yield dict(src=str(p), out=str(out_path(SOURCE, gen, p.stem)), label="spoof", source=SOURCE,
                   generator=gen, speaker="", text_id=p.stem, codec_orig="wav", license=LICENSE)


if __name__ == "__main__":
    run(jobs(), SOURCE)
