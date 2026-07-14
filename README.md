# 📄 AI-Powered Resume Information Extractor

Extract structured, validated data from PDF resumes using **Qwen2.5-7B-Instruct**, **LangChain**, **Pydantic**, and **Streamlit** — fully open-source, no OpenAI API key required.

Upload a resume, and the app parses it into a strict, typed schema covering personal details, professional summary, skills, education, work experience, projects, certifications, and more — ready to review in the UI or export as JSON, CSV, or a text report.

---

## ✨ Features

- **PDF text extraction** via `pdfplumber`, with graceful handling of encrypted, corrupted, or scanned (image-only) files.
- **Structured extraction** with Qwen2.5-7B-Instruct through LangChain's `PydanticOutputParser`.
- **Strict schema validation** with Pydantic v2 — guarantees consistent, predictable output shape.
- **Automatic cleanup + retry** for cases where the model wraps JSON in markdown fences or adds stray text.
- **Clean, sectioned Streamlit UI** for personal info, skills, education, experience, projects, certifications, and more.
- **Export options:** JSON, CSV, and a formatted plain-text report.
- **Two inference modes:** run the model **locally** (Transformers) or remotely via the **Hugging Face Inference API**.
- **Automatic GPU (CUDA) detection**, with CPU fallback.
- **Model caching** — loaded once per session via `st.cache_resource`, not reloaded on every click.
- **Robust error handling** throughout the pipeline, with clear, human-readable messages.

---

## 🗂️ Project Structure

```
resume_extractor/
├── app.py                      # Streamlit UI — orchestrates the app, no business logic
├── llm.py                      # Model loading (local Transformers or HF Inference API)
├── extractor.py                 # Orchestrates PDF -> LLM -> validated data pipeline
├── models.py                    # Pydantic schemas describing the resume data shape
├── pdf_reader.py                 # PDF text extraction + error handling
├── parser.py                     # Prompt template + PydanticOutputParser + retry logic
├── utils.py                      # Download helpers (JSON/CSV/report), validation, logging
├── checking_model_support.py     # Utility script: check HF Inference API support for a model
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Installation

### 1. Clone / copy the project

```bash
cd resume_extractor
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Activate it:
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** install the CUDA-enabled build of PyTorch that matches your CUDA version from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) instead of the default `torch` pip package, for significantly faster inference.

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Description |
|---|---|
| `INFERENCE_MODE` | `local` — runs Qwen2.5-7B-Instruct on your own machine (needs a GPU with ~16GB VRAM for smooth fp16 inference, or a slower CPU run). `api` — calls the Hugging Face Inference API instead. |
| `HF_TOKEN` | Required only when `INFERENCE_MODE=api`. Get one free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). |
| `MODEL_NAME` | Optional override for the base model repo id (defaults to `Qwen/Qwen2.5-7B-Instruct`). |

### 5. Download the model (local mode only)

The first time you run the app in local mode, `transformers` will automatically download Qwen2.5-7B-Instruct (~15GB) from Hugging Face and cache it under `~/.cache/huggingface`. This can take a while depending on your connection.

You can also pre-download a smaller variant manually for testing:

```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')"
```

To check whether a given model is currently supported ("warm") on the free HF Inference API before switching to `api` mode:

```bash
python checking_model_support.py
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## 🚀 Usage

1. Upload a PDF resume.
2. Click **Extract Information**.
3. Wait for the spinner — the model loads once, then reuses the cached instance across runs.
4. Review the structured results in each section (personal info, skills, education, experience, projects, etc.).
5. Download the results as **JSON**, **CSV**, or a formatted **text report**.
6. Click **Clear Results** to start over.

---

## 🧩 How It Works

1. **`pdf_reader.py`** extracts raw text from the uploaded PDF, page by page, and raises a clear error for encrypted, corrupted, or image-only files.
2. **`parser.py`** builds a strict extraction prompt (with a JSON schema derived from `models.py`) and sends it to the LLM.
3. The model's raw text output is cleaned (markdown fences stripped, JSON block isolated) and validated against the `ResumeData` Pydantic schema. If parsing fails, it's retried automatically.
4. **`extractor.py`** ties the PDF and LLM steps together into a single pipeline call used by the UI.
5. **`app.py`** renders the validated data in a sectioned Streamlit layout and offers JSON/CSV/text-report downloads via **`utils.py`**.

---

## 🛠️ Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `CUDA out of memory` | Model too large for your GPU. Try `INFERENCE_MODE=api`, run on CPU (slower), or use a quantized model variant. |
| Model download very slow / fails | Check your internet connection; try again — Hugging Face downloads can resume. |
| "No extractable text found" | The PDF is likely a scanned image with no text layer. Try an OCR'd or text-based PDF. |
| "password-protected/encrypted" error | Remove the password from the PDF before uploading. |
| JSON parsing keeps failing | Increase `max_new_tokens` in `llm.py`, or shorten very long resumes. The app retries once automatically. |
| `HF_TOKEN environment variable is required` | Set `INFERENCE_MODE=local`, or add a valid token to `.env` for API mode. |
| App reloads the model every click | Make sure you're not editing `app.py` mid-session — `st.cache_resource` clears on code changes, which is expected in dev. |

---

## 🧭 Roadmap / Future Improvements

- Quantized model options (4-bit/8-bit) for lower-memory local inference.
- Batch-process multiple resumes at once.
- Comparison view for ranking multiple candidates.
- Support DOCX resumes in addition to PDF.
- Unit tests for `parser.py` and `pdf_reader.py`.
- Streaming token-by-token display of model output during extraction.

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Qwen2.5-7B-Instruct (Hugging Face) |
| Orchestration | LangChain (`langchain-huggingface`) |
| Inference | Transformers `pipeline` (local) or `huggingface_hub.InferenceClient` (API) |
| PDF parsing | pdfplumber |
| Validation | Pydantic v2 |
| Data processing | pandas |
| Config | python-dotenv |

---

## 📄 License

Add your license of choice here (e.g. MIT).
"# resume_info_extractor" 
