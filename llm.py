"""
llm.py
------
Loads Qwen2.5-7B-Instruct and wraps it so LangChain can use it.

Two backends are supported (chosen via the INFERENCE_MODE env var):

1. "local"  (default) - Runs the model on your own machine with the
   Hugging Face `transformers` library (AutoTokenizer + AutoModelForCausalLM
   + pipeline). Automatically uses CUDA if a GPU is available, else CPU.
   No internet/API key needed after the model is downloaded once.

2. "api"    - Uses the Hugging Face Inference API via `huggingface_hub`'s
   InferenceClient (chat.completions style), which calls the model
   remotely. Requires an HF_TOKEN. Useful if you don't have a GPU
   locally.

Beginner note: `st.cache_resource` (used in app.py) makes sure we only
build/load this expensive object ONCE per app session, instead of
re-loading the model every time the user clicks a button.
"""

import os
import logging
from typing import Callable, Optional

import torch
from langchain_huggingface import HuggingFacePipeline

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Signature: callback(downloaded_bytes: int, total_bytes: int, filename: str) -> None
ProgressCallback = Optional[Callable[[int, int, str], None]]


def _make_progress_tqdm_class(callback: ProgressCallback):
    """
    Build a tqdm subclass that forwards progress (bytes done, bytes total,
    filename) to `callback` on every update. huggingface_hub's
    `snapshot_download` accepts a `tqdm_class` and instantiates it once
    per file being downloaded, so this is how we plug UI progress bars
    (e.g. Streamlit) into an otherwise silent download.
    """
    import tqdm as tqdm_module

    # -----------------------------------------------------------------
    # Subclass tqdm so every progress update also calls our callback
    # -----------------------------------------------------------------
    class _CallbackTqdm(tqdm_module.tqdm):
        def update(self, n=1):
            # ---------------------------------------------------------
            # Let tqdm do its normal bookkeeping first
            # ---------------------------------------------------------
            result = super().update(n)
            if callback:
                try:
                    # -----------------------------------------------------
                    # Forward current progress (n done, total, filename)
                    # -----------------------------------------------------
                    callback(self.n, self.total or 0, self.desc or "")
                except Exception:  # noqa: BLE001
                    # Never let a UI callback crash the download.
                    # -------------------------------------------------
                    # Swallow callback errors so the download continues
                    # -------------------------------------------------
                    logger.debug("Progress callback raised", exc_info=True)
            return result

    return _CallbackTqdm


def download_model_with_progress(model_name: str, progress_callback: ProgressCallback = None) -> str:
    """
    Pre-download the model repo (weights, tokenizer, config) via
    huggingface_hub, reporting per-file byte progress through
    `progress_callback`. If the model is already cached, files with
    nothing left to fetch simply report 100% almost instantly.

    Returns the local snapshot path (not required by callers, but
    useful for debugging/logging).
    """
    from huggingface_hub import snapshot_download

    # ---------------------------------------------------------------------
    # Build a progress-reporting tqdm class only if a callback was given
    # ---------------------------------------------------------------------
    tqdm_class = _make_progress_tqdm_class(progress_callback) if progress_callback else None
    logger.info("Ensuring model '%s' is downloaded...", model_name)
    # ---------------------------------------------------------------------
    # Download (or confirm cached) model snapshot from the Hub
    # ---------------------------------------------------------------------
    local_path = snapshot_download(repo_id=model_name, tqdm_class=tqdm_class)
    logger.info("Model '%s' available at '%s'.", model_name, local_path)
    return local_path

# Generation settings tuned for deterministic, structured JSON output.
GENERATION_KWARGS = {
    "temperature": 0.1,
    "do_sample": False,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.05,
    "return_full_text": False,
}


def _select_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    # ---------------------------------------------------------------------
    # Auto-detect GPU availability and pick device accordingly
    # ---------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Selected inference device: %s", device)
    return device


def load_local_llm(model_name: str = DEFAULT_MODEL_NAME, progress_callback: ProgressCallback = None) -> HuggingFacePipeline:
    """
    Load Qwen2.5-7B-Instruct locally using Transformers and wrap it in
    a LangChain-compatible HuggingFacePipeline.

    Args:
        model_name: Hugging Face model repo id.
        progress_callback: optional fn(downloaded_bytes, total_bytes, filename)
            called during the initial download, e.g. to drive a UI
            progress bar. No-op once files are already cached.

    Returns:
        A HuggingFacePipeline instance ready to use inside LangChain.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    # ---------------------------------------------------------------------
    # Pick CPU or CUDA before loading any heavy objects
    # ---------------------------------------------------------------------
    device = _select_device()

    # Downloads (or confirms cache) before the slow from_pretrained calls,
    # so callers can show real download progress instead of a blind spinner.
    # ---------------------------------------------------------------------
    # Ensure model files are present locally before loading them
    # ---------------------------------------------------------------------
    download_model_with_progress(model_name, progress_callback)

    logger.info("Loading tokenizer and model '%s' (this can take a while)...", model_name)

    # ---------------------------------------------------------------------
    # Load tokenizer and model weights from the local cache
    # ---------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        # -------------------------------------------------------------
        # device_map="auto" only applies on CUDA; explicitly move to CPU
        # -------------------------------------------------------------
        model.to("cpu")

    # ---------------------------------------------------------------------
    # Wrap model + tokenizer into a text-generation pipeline
    # ---------------------------------------------------------------------
    text_gen_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        **GENERATION_KWARGS,
    )

    # ---------------------------------------------------------------------
    # Wrap the raw pipeline in a LangChain-compatible interface
    # ---------------------------------------------------------------------
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    logger.info("Local LLM pipeline ready.")
    return llm


class HFInferenceAPILLM:
    """
    Minimal LangChain-Runnable-like wrapper around huggingface_hub's
    InferenceClient (remote chat-completions API), so it can be used
    with `.invoke(prompt)` the same way a LangChain LLM is used.

    This mirrors the style of:

        from huggingface_hub import InferenceClient
        client = InferenceClient(api_key=os.environ["HF_TOKEN"])
        client.chat.completions.create(model=..., messages=[...])
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME + ":together", api_key: Optional[str] = None):
        from huggingface_hub import InferenceClient

        # -------------------------------------------------------------
        # Resolve API key from arg or environment; fail fast if missing
        # -------------------------------------------------------------
        api_key = api_key or os.environ.get("HF_TOKEN")
        if not api_key:
            raise ValueError(
                "HF_TOKEN environment variable is required for INFERENCE_MODE=api. "
                "Set it in your .env file."
            )
        self.model_name = model_name
        # -------------------------------------------------------------
        # Create the remote inference client used by .invoke()
        # -------------------------------------------------------------
        self.client = InferenceClient(api_key=api_key)

    def invoke(self, prompt: str) -> str:
        """Send a prompt to the HF Inference API and return the full text response."""
        # ---------------------------------------------------------------
        # Call the remote chat-completions endpoint with a single user turn
        # ---------------------------------------------------------------
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=GENERATION_KWARGS["temperature"],
            max_tokens=GENERATION_KWARGS["max_new_tokens"],
        )
        # ---------------------------------------------------------------
        # Extract plain text content from the first completion choice
        # ---------------------------------------------------------------
        return response.choices[0].message.content


def load_llm(progress_callback: ProgressCallback = None):
    """
    Factory function: reads INFERENCE_MODE from the environment and
    returns either a local HuggingFacePipeline or an HFInferenceAPILLM.

    Set in .env:
        INFERENCE_MODE=local   (default, runs on your machine)
        INFERENCE_MODE=api     (uses HF Inference API, needs HF_TOKEN)

    Args:
        progress_callback: only used in local mode; forwarded to the
            model download step to report byte-level progress.
    """
    # ---------------------------------------------------------------------
    # Read backend selection + model name from environment (with defaults)
    # ---------------------------------------------------------------------
    mode = os.environ.get("INFERENCE_MODE", "local").lower()
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)

    logger.info("Initializing LLM in '%s' mode.", mode)

    if mode == "api":
        # -------------------------------------------------------------
        # Remote inference: build the HF Inference API wrapper
        # -------------------------------------------------------------
        api_model_name = os.environ.get("API_MODEL_NAME", f"{model_name}:together")
        return HFInferenceAPILLM(model_name=api_model_name)

    # ---------------------------------------------------------------------
    # Local inference: load model + tokenizer on this machine (default path)
    # ---------------------------------------------------------------------
    return load_local_llm(model_name=model_name, progress_callback=progress_callback)
