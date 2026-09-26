"""Full LJSpeech-1.1 (Ito & Johnson 2017, public domain) -> data/processed/ljspeech/.

Streams the 2.6 GB tar.bz2 straight into canonical clips (no raw copy on disk).
Skips the 242 ids already given in data/raw/lj_real so the two sources never overlap.
"""
import tarfile
import urllib.request
from pathlib import PurePosixPath

from ingest_common import DATA, out_path, run

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
SOURCE = "ljspeech"


def jobs():
    given = {p.stem for p in (DATA / "raw" / "lj_real").glob("*.wav")}
    assert len(given) == 242, len(given)
    with urllib.request.urlopen(URL, timeout=300) as resp, tarfile.open(fileobj=resp, mode="r|bz2") as tar:
        for m in tar:
            name = PurePosixPath(m.name)
            if not (m.isfile() and name.suffix == ".wav") or name.stem in given:
                continue
            yield dict(src=tar.extractfile(m).read(), out=str(out_path(SOURCE, name.stem)), label="bonafide",
                       source=SOURCE, generator="bonafide", speaker="LJ", text_id=name.stem,
                       codec_orig="wav", license="Public domain (LJSpeech-1.1)")


if __name__ == "__main__":
    run(jobs(), SOURCE)
