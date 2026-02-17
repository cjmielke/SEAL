from pathlib import Path
from typing import Optional, Union

from huggingface_hub import snapshot_download
from huggingface_hub.utils import (
    are_progress_bars_disabled,
    disable_progress_bars,
    enable_progress_bars,
)


def _required_hest_files(sample_id: str):
    return [
        f"st/{sample_id}.h5ad",
        f"wsis/{sample_id}.tif",
        f"metadata/{sample_id}.json",
    ]


def _optional_hest_patterns(sample_id: str):
    return [
        f"tissue_seg/*{sample_id}*",
        f"cellvit_seg/*{sample_id}*",
        f"xenium_seg/*{sample_id}*",
        f"transcripts/{sample_id}_transcripts.parquet",
        f"patches_mpp_025/{sample_id}.h5",
        f"patches_mpp_025/{sample_id}_patch_vis.png",
    ]


def download_hest_sample(
    sample_id: str,
    local_dir: Union[str, Path],
    repo_id: str = "MahmoodLab/hest",
    token: Optional[str] = None,
) -> Path:
    """Download (or reuse cached) sample-scoped HEST files and return local HEST root."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    required = _required_hest_files(sample_id)
    required_paths = [local_dir / rel for rel in required]

    if all(p.exists() for p in required_paths):
        print("Using cached HEST subset from:", local_dir)
        return local_dir

    allow_patterns = required + _optional_hest_patterns(sample_id) + ["README.md", "HEST_v1_1_0.csv"]

    # In some notebook environments tqdm.notebook can fail at teardown
    # ("tqdm object has no attribute 'disp'"). Disable HF progress bars
    # for this call to avoid that path.
    prev_disabled = are_progress_bars_disabled()
    disable_progress_bars()
    try:
        resolved = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
            token=token,
        )
    finally:
        if not prev_disabled:
            enable_progress_bars()

    resolved = Path(resolved)
    print("Downloaded/resolved HEST subset to:", resolved)
    return resolved
