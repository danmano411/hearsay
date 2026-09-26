"""WaveFake (Frank & Schoenherr, NeurIPS 2021 D&B) -> data/processed/wavefake/.

LJ-voice fakes: 7 vocoders re-synthesising LJSpeech + a Conformer/FastSpeech2 TTS in the LJ voice.
The 28.9 GB Zenodo zip is never downloaded: we read its central directory and fetch only sampled members
with HTTP Range requests (~10% of the archive).
"""
import itertools
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

from ingest_common import HttpRangeFile, out_path, run, zip_member_bytes

URL = "https://zenodo.org/records/5642694/files/generated_audio.zip"
SOURCE = "wavefake"
LICENSE = "CC-BY-SA-4.0"
PER_GENERATOR = {  # LJSpeech-based subsets only (JSUT is Japanese)
    "ljspeech_melgan": 1500, "ljspeech_melgan_large": 1500, "ljspeech_full_band_melgan": 1500,
    "ljspeech_multi_band_melgan": 1500, "ljspeech_parallel_wavegan": 1500, "ljspeech_waveglow": 1500,
    "ljspeech_hifiGAN": 1500, "common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech": 2000,
}


def jobs():
    remote = HttpRangeFile(URL)
    infos = [i for i in zipfile.ZipFile(remote).infolist() if i.filename.endswith(".wav")]
    rng = random.Random(0)
    groups = []
    for gen, n in PER_GENERATOR.items():
        # Zenodo rate-limits per request (HTTP 429), so take a block of members that is contiguous in the archive
        # (member order inside each folder is unrelated to LJ ids, so a block is still a scattered id sample)
        # and fetch it in ~40 MB ranged reads instead of one request per file.
        members = sorted((i for i in infos if i.filename.split("/")[1] == gen), key=lambda i: i.header_offset)
        start = rng.randrange(max(1, len(members) - n))
        block = members[start:start + n]
        for k in range(0, len(block), 120):
            groups.append((gen, block[k:k + 120]))

    def fetch(group):
        gen, members = group
        todo = [i for i in members if not out_path(SOURCE, gen, PurePosixPath(i.filename).stem).exists()]
        buf, base = b"", 0
        if todo:
            base = todo[0].header_offset
            last = todo[-1]
            buf = remote.read_range(base, last.header_offset + last.compress_size + 2048 - base)
        jobs = []
        for info in members:
            stem = PurePosixPath(info.filename).stem  # e.g. LJ001-0001_gen
            out = out_path(SOURCE, gen, stem)
            src = zip_member_bytes(lambda o, n: buf[o - base:o - base + n], info) if info in todo else None
            jobs.append(dict(src=src, out=str(out), label="spoof", source=SOURCE, generator=f"wavefake_{gen}",
                             speaker="LJ", text_id=stem.removesuffix("_gen"), sr_orig=22050, codec_orig="wav",
                             license=LICENSE))
        return jobs

    with ThreadPoolExecutor(3) as tp:
        for chunk in itertools.batched(groups, 3):
            for jobs_ in tp.map(fetch, chunk):
                yield from jobs_


if __name__ == "__main__":
    run(jobs(), SOURCE)
