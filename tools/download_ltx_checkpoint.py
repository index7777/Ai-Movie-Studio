"""Download only the first LTX feasibility checkpoint."""

from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "Lightricks/LTX-Video"
FILENAME = "ltxv-2b-0.9.6-distilled-04-25.safetensors"


def main() -> None:
    target = Path("models/ltx")
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {FILENAME} to {target.resolve()}")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=target,
        repo_type="model",
    )
    print(f"CHECKPOINT: {path}")


if __name__ == "__main__":
    main()
