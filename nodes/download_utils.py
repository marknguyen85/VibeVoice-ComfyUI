import logging
import os

logger = logging.getLogger("VibeVoice")

TOKENIZER_REPO = "Qwen/Qwen2.5-1.5B"
TOKENIZER_FILES = ["tokenizer_config.json", "vocab.json", "merges.txt", "tokenizer.json"]

# Known HuggingFace repos — key is the local folder name shown in dropdown
KNOWN_MODEL_REPOS = {
    "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
    "VibeVoice-Large": "aoi-ot/VibeVoice-Large",
}

# Approximate total sizes for user info
_MODEL_SIZES = {
    "VibeVoice-1.5B": "~5.4 GB",
    "VibeVoice-Large": "~18.7 GB",
}


def _hf_hub_available() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _download_file_with_log(repo_id: str, filename: str, local_dir: str, idx: int, total: int):
    """Download a single file from HF and log before/after with size."""
    from huggingface_hub import hf_hub_download, get_hf_file_metadata, hf_hub_url

    # Try to get file size before downloading
    size_str = ""
    try:
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        meta = get_hf_file_metadata(url)
        if meta.size:
            size_str = f" ({_format_bytes(meta.size)})"
    except Exception:
        pass

    logger.info(f"  [{idx}/{total}] Downloading {filename}{size_str} ...")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

    # Log actual file size from disk
    try:
        actual = os.path.getsize(local_path)
        logger.info(f"  [{idx}/{total}] Done: {filename} ({_format_bytes(actual)})")
    except Exception:
        logger.info(f"  [{idx}/{total}] Done: {filename}")


def ensure_tokenizer(vibevoice_dir: str) -> str | None:
    """Download Qwen tokenizer to vibevoice_dir/tokenizer/ if not already present.

    Returns the tokenizer directory path on success, None on failure.
    """
    tokenizer_dir = os.path.join(vibevoice_dir, "tokenizer")
    required = ["tokenizer_config.json", "vocab.json", "merges.txt"]

    if all(os.path.exists(os.path.join(tokenizer_dir, f)) for f in required):
        return tokenizer_dir

    if not _hf_hub_available():
        logger.error("huggingface_hub not installed — cannot auto-download tokenizer. "
                     "Run: pip install huggingface_hub")
        return None

    logger.info(f"Tokenizer not found — downloading {TOKENIZER_REPO} ({len(TOKENIZER_FILES)} files) ...")
    os.makedirs(tokenizer_dir, exist_ok=True)

    try:
        total = len(TOKENIZER_FILES)
        for idx, filename in enumerate(TOKENIZER_FILES, 1):
            try:
                _download_file_with_log(TOKENIZER_REPO, filename, tokenizer_dir, idx, total)
            except Exception as e:
                if filename in required:
                    raise
                logger.warning(f"  [{idx}/{total}] Optional file '{filename}' skipped: {e}")

        logger.info("Tokenizer download complete.")
        return tokenizer_dir

    except Exception as e:
        logger.error(f"Failed to download tokenizer: {e}")
        return None


def ensure_model(model_name: str, vibevoice_dir: str) -> str | None:
    """Download a known VibeVoice model from HuggingFace file-by-file with progress logging.

    Returns the local model directory on success, None on failure.
    """
    repo_id = KNOWN_MODEL_REPOS.get(model_name)
    if not repo_id:
        return None

    if not _hf_hub_available():
        logger.error("huggingface_hub not installed — cannot auto-download model. "
                     "Run: pip install huggingface_hub")
        return None

    model_dir = os.path.join(vibevoice_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    size_hint = _MODEL_SIZES.get(model_name, "large")
    logger.info(f"Model '{model_name}' not found locally.")
    logger.info(f"Downloading from {repo_id} ({size_hint}) — this may take a while ...")

    try:
        from huggingface_hub import list_repo_files

        # Get full file list so we can show N/total
        all_files = list(list_repo_files(repo_id))
        total = len(all_files)
        logger.info(f"Repository has {total} file(s) to download.")

        skipped = 0
        for idx, filename in enumerate(all_files, 1):
            local_path = os.path.join(model_dir, filename)
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                skipped += 1
                logger.info(f"  [{idx}/{total}] Skipping (already exists): {filename}")
                continue
            try:
                _download_file_with_log(repo_id, filename, model_dir, idx, total)
            except Exception as e:
                logger.error(f"  [{idx}/{total}] Failed to download '{filename}': {e}")
                raise

        logger.info(f"Model '{model_name}' download complete "
                    f"({total - skipped} downloaded, {skipped} already present).")
        return model_dir

    except Exception as e:
        logger.error(f"Failed to download model '{model_name}': {e}")
        return None
