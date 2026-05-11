import logging
import os
import pickle
import re
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax.training import checkpoints

from utils.logging_utils import log_for_0


# ============================================
# Save
# ============================================

def save_checkpoint(state: Any, output_dir: str, step: int, epoch_name: str = None):
    """Save model checkpoint to a local directory."""
    state_unreplicated = jax_utils.unreplicate(state)

    state_dict = {
        "params": jax.tree_util.tree_map(np.array, state_unreplicated.params),
        "ema_params1": jax.tree_util.tree_map(np.array, state_unreplicated.ema_params1),
        "opt_state": jax.tree_util.tree_map(
            lambda x: np.array(x) if hasattr(x, "shape") else x,
            state_unreplicated.opt_state,
        ),
        "step": int(state_unreplicated.step),
        "epoch": int(state_unreplicated.epoch),
        "dropout_rng": np.array(state_unreplicated.dropout_rng),
    }

    loggers_to_suppress = ["flax.training.checkpoints", "absl", "orbax", "tensorstore"]
    old_levels = {name: logging.getLogger(name).level for name in loggers_to_suppress}
    for name in loggers_to_suppress:
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        ckpt_dir = os.path.abspath(output_dir)
        log_for_0(f"Saving checkpoint to {ckpt_dir}")
        if epoch_name:
            checkpoints.save_checkpoint_multiprocess(
                ckpt_dir, state_dict, 0,
                prefix=f"checkpoint_{epoch_name}_", keep=1, overwrite=True,
            )
        else:
            checkpoints.save_checkpoint_multiprocess(
                ckpt_dir, state_dict, step, keep=10, overwrite=True,
            )
    finally:
        for name, level in old_levels.items():
            logging.getLogger(name).setLevel(level)


# ============================================
# Encoder checkpoint (single pickle file)
# ============================================

def load_encoder_checkpoint(checkpoint_path: str):
    """Load a pickled encoder checkpoint from a local path or HF Hub.

    HF form: '<org>/<repo>/<filename>'.
    """
    if not checkpoint_path:
        raise ValueError(
            "encoder_checkpoint is not set. Provide a local path or HF Hub path "
            "like 'embedded-language-flows/t5_small_encoder_jax/t5_small_encoder_jax.pkl'."
        )

    log_for_0(f"Loading encoder checkpoint from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = _download_hf_file(checkpoint_path)
    with open(checkpoint_path, "rb") as f:
        loaded_params = pickle.load(f)

    if isinstance(loaded_params, dict) and "params" in loaded_params:
        return loaded_params["params"]
    return loaded_params


def _download_hf_file(path: str) -> str:
    """Download a single file from HF Hub and return its local cache path."""
    if os.path.isabs(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    hf_path = path[5:] if path.startswith("hf://") else path
    parts = hf_path.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid HF path {path!r}. Expected '<org>/<repo>/<filename>'."
        )
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])

    try:
        from huggingface_hub import hf_hub_download

        log_for_0(f"Checkpoint not found locally, downloading from HF: {repo_id}/{filename}")
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    except Exception as e:
        raise FileNotFoundError(f"Checkpoint not found locally or on HF: {path} ({e})") from e


# ============================================
# Path helpers: local dir vs HF repo
# ============================================

def _is_hf_repo_path(path: str) -> bool:
    """Heuristic: treat as HF '<org>/<repo>[/sub/path]' if not a local path."""
    if path.startswith("/") or path.startswith(".") or path.startswith("~"):
        return False
    if os.path.isdir(path) or os.path.isfile(path):
        return False
    return len(path.split("/")) >= 2


def _hf_repo_and_subdir(path: str) -> Tuple[str, str]:
    """Split '<org>/<repo>[/sub/path]' into (repo_id, sub_path)."""
    parts = path.split("/")
    return "/".join(parts[:2]), "/".join(parts[2:])


def _checkpoint_step(checkpoint_name: str) -> int:
    """Extract the trailing checkpoint step from a name; -1 if absent."""
    match = re.search(r"(\d+)$", checkpoint_name)
    return int(match.group(1)) if match else -1


# ============================================
# Resume: list + load (local or HF)
# ============================================

def find_all_checkpoints(ckpt_dir: str, prefix: str = "checkpoint_"):
    """Find all checkpoint paths in a dir, sorted by step ascending.

    `ckpt_dir` may be a local directory or an HF repo id ('<org>/<repo>').
    Returns absolute local paths for local dirs, or '<org>/<repo>/<name>' for HF.
    """
    if _is_hf_repo_path(ckpt_dir):
        from huggingface_hub import HfApi
        repo_id, _ = _hf_repo_and_subdir(ckpt_dir)
        api = HfApi()
        if not api.repo_exists(repo_id, repo_type="model"):
            return []
        files = api.list_repo_files(repo_id, repo_type="model")
        names = sorted(
            {f.split("/")[0] for f in files if f.startswith(prefix)},
            key=_checkpoint_step,
        )
        return [f"{repo_id}/{name}" for name in names]

    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        return []
    names = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith(prefix)],
        key=_checkpoint_step,
    )
    return [os.path.join(ckpt_dir, name) for name in names]


def find_latest_checkpoint(ckpt_dir: str, prefix: str = "checkpoint_"):
    """Return the latest checkpoint path (local or HF), or None."""
    all_ckpts = find_all_checkpoints(ckpt_dir, prefix)
    return all_ckpts[-1] if all_ckpts else None


def _resolve_local_ckpt(ckpt_path: str) -> str:
    """Return a local absolute path. Downloads HF snapshot if `ckpt_path` is an HF repo path."""
    if not _is_hf_repo_path(ckpt_path):
        return os.path.abspath(ckpt_path)
    from huggingface_hub import snapshot_download
    repo_id, sub_path = _hf_repo_and_subdir(ckpt_path)
    log_for_0(f"Downloading checkpoint from HF: {repo_id}" + (f"/{sub_path}" if sub_path else ""))
    local_dir = snapshot_download(
        repo_id=repo_id, repo_type="model",
        allow_patterns=[f"{sub_path}/**"] if sub_path else None,
    )
    return os.path.join(local_dir, sub_path) if sub_path else local_dir


def load_checkpoint(checkpoint_path: str, state_template: Any) -> Tuple[Any, int]:
    """Load an ELF checkpoint (local path or HF '<org>/<repo>/checkpoint_<step>')."""
    log_for_0(f"Loading ELF checkpoint from {checkpoint_path}...")

    target = {
        "params": state_template.params,
        "ema_params1": state_template.ema_params1,
        "opt_state": state_template.opt_state,
        "step": state_template.step,
        "epoch": state_template.epoch,
        "dropout_rng": state_template.dropout_rng,
    }
    checkpoint_path = _resolve_local_ckpt(checkpoint_path)
    log_for_0(f"Loading checkpoint from {checkpoint_path}...")

    loggers_to_suppress = ["flax.training.checkpoints", "absl", "orbax", "tensorstore"]
    old_levels = {name: logging.getLogger(name).level for name in loggers_to_suppress}
    for name in loggers_to_suppress:
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        ckpt = checkpoints.restore_checkpoint(checkpoint_path, target=target)
    finally:
        for name, level in old_levels.items():
            logging.getLogger(name).setLevel(level)
    log_for_0(f"Loaded checkpoint keys: {ckpt.keys() if ckpt else 'None'}")

    if ckpt is None or int(ckpt.get("step", 0)) == 0:
        raise ValueError(f"Failed to load checkpoint from {checkpoint_path}")

    restored_state = state_template.replace(
        params=jax.tree_util.tree_map(jnp.array, ckpt["params"]),
        ema_params1=jax.tree_util.tree_map(jnp.array, ckpt.get("ema_params1", ckpt["params"])),
        opt_state=ckpt["opt_state"],
        step=ckpt["step"],
        epoch=ckpt["epoch"],
        dropout_rng=jnp.array(ckpt["dropout_rng"]),
    )
    step, epoch = int(ckpt["step"]), int(ckpt["epoch"])
    log_for_0(f"Loaded checkpoint from step {step} (epoch {epoch})")
    return restored_state, step
