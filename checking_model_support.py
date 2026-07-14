# ---------------------------------------------------------------------
# Check whether a given Hugging Face model is currently supported
# ("warm") on the free Inference API.
# ---------------------------------------------------------------------
from huggingface_hub import model_info

# ---------------------------------------------------------------------
# Fetch model metadata, including inference-status info, from the Hub
# ---------------------------------------------------------------------
info = model_info("Qwen/Qwen2.5-7B-Instruct", expand="inference")

# ---------------------------------------------------------------------
# Print the inference status: "warm" = supported, None = not supported
# ---------------------------------------------------------------------
print(info.inference)   # "warm" = supported, None = not
