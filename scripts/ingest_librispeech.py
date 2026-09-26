"""LibriSpeech (Panayotov et al. 2015, CC BY 4.0) -> data/processed/librispeech/.

* All 10 speakers DiffSSD clones (100, 1487, ... 8848). Per SPEAKERS.TXT they are all in train-clean-360
  (not train-clean-100), so we stream that 23 GB tarball and keep every one of their utterances:
  matched real/fake pairs for the cloning generators.
* Speaker diversity: every other train-clean-360 speaker contributes its first OTHER_PER_SPEAKER utterances,
  plus all of dev-clean and test-clean (80 more speakers).
Tarballs are streamed; nothing raw is stored.
"""
import tarfile
import urllib.request
from collections import Counter
from pathlib import PurePosixPath

from ingest_common import out_path, run

BASE = "https://www.openslr.org/resources/12/"
SOURCE = "librispeech"
DIFFSSD_SPEAKERS = {"100", "1487", "2061", "3654", "4490", "5448", "6167", "6575", "7995", "8848"}
OTHER_PER_SPEAKER = 12


def jobs():
    for subset in ["dev-clean", "test-clean", "train-clean-360"]:
        per_spk = Counter()
        with urllib.request.urlopen(BASE + subset + ".tar.gz", timeout=300) as resp, \
                tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for m in tar:
                name = PurePosixPath(m.name)
                if not (m.isfile() and name.suffix == ".flac"):
                    continue
                spk = name.stem.split("-")[0]
                if subset == "train-clean-360" and spk not in DIFFSSD_SPEAKERS:
                    per_spk[spk] += 1
                    if per_spk[spk] > OTHER_PER_SPEAKER:
                        continue
                yield dict(src=tar.extractfile(m).read(), out=str(out_path(SOURCE, subset, spk, name.stem)),
                           label="bonafide", source=SOURCE, generator="bonafide", speaker=spk,
                           text_id=name.stem, codec_orig="flac", license="CC-BY-4.0")


if __name__ == "__main__":
    run(jobs(), SOURCE)
