"""--device handling shared by the torch scripts: 'auto' (default) = cuda if available, else cpu."""
import torch

CUDA_MEMORY_FRACTION = 0.92


def resolve(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {name} requested but torch.cuda.is_available() is False "
                         f"(torch {torch.__version__}: install a CUDA build, see docs/GPU_HANDOFF.md)")
    dev = torch.device(name)
    if dev.type == "cuda":
        # Windows (WDDM) lets an over-allocation spill into shared system memory, which runs ~100x slower instead of
        # failing (an 8 GB card at WavLM batch 64 crawled for 20+ min). Capping the caching allocator below the card
        # size turns that into a clean torch.OutOfMemoryError the caller can catch (smaller batch) or see at once.
        # needs an indexed device: torch.device("cuda") (what --device auto/cuda gives) raises ValueError
        torch.cuda.set_per_process_memory_fraction(
            CUDA_MEMORY_FRACTION, dev.index if dev.index is not None else torch.cuda.current_device())
    return dev


def add_argument(parser):
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N (auto = cuda if available)")
