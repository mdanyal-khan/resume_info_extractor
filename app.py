import logging

import streamlit as st
from dotenv import load_dotenv

from llm import load_llm
from extractor import run_extraction_pipeline
from pdf_reader import PDFReadError
from utils import (
    setup_logging,
    validate_uploaded_file,
    resume_to_json_bytes,
    resume_to_csv_bytes,
    resume_to_text_report,
)

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
load_dotenv()
logger = setup_logging()

st.set_page_config(
    page_title="AI Resume Extractor",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_llm(_progress_callback=None):
    """
    Load the LLM exactly once per app session and cache it.
    st.cache_resource is designed for objects like models/DB connections
    that are expensive to create and safe to reuse across reruns.

    Note: the leading underscore on `_progress_callback` tells Streamlit
    to skip hashing that argument (functions aren't hashable) without
    breaking caching.
    """
    logger.info("Loading LLM (this only happens once per session)...")
    # ---------------------------------------------------------------------
    # Delegate actual model loading to llm.py; result gets cached by
    # st.cache_resource so this body only runs once per session
    # ---------------------------------------------------------------------
    return load_llm(progress_callback=_progress_callback)


def render_download_progress():
    """
    Build a Streamlit progress bar + status line, return a callback that
    updates them from llm.py's download hook: fn(downloaded, total, filename).
    """
    # ---------------------------------------------------------------------
    # Create empty progress bar + status text placeholders
    # ---------------------------------------------------------------------
    bar = st.progress(0)
    status = st.empty()

    def _update(downloaded: int, total: int, filename: str):
        # -------------------------------------------------------------
        # Compute percentage complete, guarding against total == 0
        # -------------------------------------------------------------
        if total > 0:
            pct = min(int(downloaded / total * 100), 100)
        else:
            pct = 0
        mb_done = downloaded / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        # -------------------------------------------------------------
        # Push updated values into the progress bar + status text
        # -------------------------------------------------------------
        bar.progress(pct)
        status.text(f"⬇️ {filename}: {mb_done:.1f} MB / {mb_total:.1f} MB ({pct}%)")

    return bar, status, _update


def init_session_state():
    """Initialize keys we rely on across Streamlit reruns."""
    # ---------------------------------------------------------------------
    # Default values for all session-state keys used by this app
    # ---------------------------------------------------------------------
    defaults = {
        "resume_text": None,
        "num_pages": None,
        "filename": None,
        "structured_data": None,
        "error_message": None,
    }
    # ---------------------------------------------------------------------
    # Only set keys that don't already exist (preserve state across reruns)
    # ---------------------------------------------------------------------
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_results():
    """Reset all extraction results (used by the Clear Results button)."""
    # ---------------------------------------------------------------------
    # Wipe all extraction-related session state back to empty
    # ---------------------------------------------------------------------
    st.session_state["resume_text"] = None
    st.session_state["num_pages"] = None
    st.session_state["filename"] = None
    st.session_state["structured_data"] = None
    st.session_state["error_message"] = None
    logger.info("Results cleared by user.")


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:

        # -------------------------------------------------------------
        # Instructions block
        # -------------------------------------------------------------
        st.subheader("📋 Instructions")
        st.markdown(
            """
1. Upload a resume in **PDF** format.
2. Click **Extract Information**.
3. Wait for the model to process the resume.
4. Review the structured results below.
5. Download as JSON, CSV, or a text report.
            """
        )
        st.divider()
        # -------------------------------------------------------------
        # Supported file type note
        # -------------------------------------------------------------
        st.subheader("📁 Supported File Type")
        st.markdown("- PDF (`.pdf`) only, up to 10 MB")


# ---------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------
def render_structured_results(data):
    # ---------------------------------------------------------------------
    # Convert Pydantic model to plain dict once for all sections below
    # ---------------------------------------------------------------------
    d = data.model_dump()

    # ---------------------------------------------------------------------
    # Personal information table
    # ---------------------------------------------------------------------
    st.subheader("👤 Personal Information")
    st.table({k.replace("_", " ").title(): [v or "N/A"] for k, v in d["personal_information"].items()})

    # ---------------------------------------------------------------------
    # Professional information table
    # ---------------------------------------------------------------------
    st.subheader("💼 Professional Information")
    st.table({k.replace("_", " ").title(): [v or "N/A"] for k, v in d["professional_information"].items()})

    # ---------------------------------------------------------------------
    # Professional summary text
    # ---------------------------------------------------------------------
    st.subheader("📝 Professional Summary")
    st.write(d.get("professional_summary") or "N/A")

    # ---------------------------------------------------------------------
    # Skills grid: spread each skill category across 4 columns
    # ---------------------------------------------------------------------
    st.subheader("🛠️ Skills")
    skills = d["skills"]
    cols = st.columns(4)
    skill_items = list(skills.items())
    for i, (key, values) in enumerate(skill_items):
        with cols[i % 4]:
            st.markdown(f"**{key.replace('_', ' ').title()}**")
            if values:
                for v in values:
                    st.markdown(f"- {v}")
            else:
                st.caption("None listed")

    # ---------------------------------------------------------------------
    # Education table (or empty-state caption)
    # ---------------------------------------------------------------------
    st.subheader("🎓 Education")
    if d["education"]:
        st.dataframe(d["education"], use_container_width=True)
    else:
        st.caption("No education records found.")

    # ---------------------------------------------------------------------
    # Experience: one collapsible expander per entry
    # ---------------------------------------------------------------------
    st.subheader("🏢 Experience")
    if d["experience"]:
        for exp in d["experience"]:
            with st.expander(f"{exp.get('position', 'N/A')} @ {exp.get('company', 'N/A')}"):
                st.json(exp)
    else:
        st.caption("No experience records found.")

    # ---------------------------------------------------------------------
    # Projects: one collapsible expander per entry
    # ---------------------------------------------------------------------
    st.subheader("🚀 Projects")
    if d["projects"]:
        for proj in d["projects"]:
            with st.expander(proj.get("name") or "Untitled project"):
                st.json(proj)
    else:
        st.caption("No projects found.")

    # ---------------------------------------------------------------------
    # Remaining simple-list sections, split across two columns
    # ---------------------------------------------------------------------
    simple_list_sections = [
        ("📜 Certifications", "certifications"),
        ("🗣️ Languages", "languages"),
        ("🏆 Achievements", "achievements"),
        ("🎯 Internships", "internships"),
        ("🥇 Awards", "awards"),
        ("🤝 Volunteer Experience", "volunteer_experience"),
    ]
    col_a, col_b = st.columns(2)
    for i, (title, key) in enumerate(simple_list_sections):
        target = col_a if i % 2 == 0 else col_b
        with target:
            st.markdown(f"**{title}**")
            if d[key]:
                for item in d[key]:
                    st.markdown(f"- {item}")
            else:
                st.caption("None listed")

    # ---------------------------------------------------------------------
    # Raw JSON dump for debugging/power users
    # ---------------------------------------------------------------------
    with st.expander("🔍 Raw JSON"):
        st.json(d)


def render_downloads(data, filename):
    st.subheader("⬇️ Downloads")
    col1, col2, col3 = st.columns(3)

    # ---------------------------------------------------------------------
    # Strip extension from filename to build download filenames
    # ---------------------------------------------------------------------
    base_name = filename.rsplit(".", 1)[0] if filename else "resume"

    # ---------------------------------------------------------------------
    # JSON download button
    # ---------------------------------------------------------------------
    with col1:
        st.download_button(
            "Download JSON",
            data=resume_to_json_bytes(data),
            file_name=f"{base_name}_extracted.json",
            mime="application/json",
        )
    # ---------------------------------------------------------------------
    # CSV download button
    # ---------------------------------------------------------------------
    with col2:
        st.download_button(
            "Download CSV",
            data=resume_to_csv_bytes(data),
            file_name=f"{base_name}_extracted.csv",
            mime="text/csv",
        )
    # ---------------------------------------------------------------------
    # Text report download button
    # ---------------------------------------------------------------------
    with col3:
        st.download_button(
            "Download Text Report",
            data=resume_to_text_report(data),
            file_name=f"{base_name}_report.txt",
            mime="text/plain",
        )


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------
def main():
    # ---------------------------------------------------------------------
    # Initialize session state + sidebar before rendering the main body
    # ---------------------------------------------------------------------
    init_session_state()
    render_sidebar()

    st.title("📄 AI-Powered Resume Information Extractor")
    st.markdown(
        "Upload a resume in PDF format and extract structured, validated information from it — personal details, skills, education, experience, projects, and more."
    )

    uploaded_file = st.file_uploader("Upload a resume (PDF)", type=["pdf"])

    col_extract, col_clear = st.columns([1, 1])
    extract_clicked = col_extract.button("🔎 Extract Information", type="primary", use_container_width=True)
    clear_clicked = col_clear.button("🗑️ Clear Results", use_container_width=True)

    if clear_clicked:
        # -------------------------------------------------------------
        # User clicked Clear Results -> wipe state and rerun the app
        # -------------------------------------------------------------
        clear_results()
        st.rerun()

    if extract_clicked:
        if uploaded_file is None:
            # -----------------------------------------------------------
            # Guard: nothing uploaded yet
            # -----------------------------------------------------------
            st.error("Please upload a PDF file first.")
        else:
            try:
                # ---------------------------------------------------------
                # Validate file type/size before doing any real work
                # ---------------------------------------------------------
                validate_uploaded_file(uploaded_file.name, uploaded_file.size)
                logger.info("File uploaded: %s (%d bytes)", uploaded_file.name, uploaded_file.size)

                st.caption("Loading model")
                # ---------------------------------------------------------
                # Show download progress UI while the (cached) LLM loads
                # ---------------------------------------------------------
                bar, status, update_progress = render_download_progress()
                try:
                    llm = get_llm(update_progress)
                finally:
                    # -------------------------------------------------
                    # Always clear progress widgets, even on failure
                    # -------------------------------------------------
                    bar.empty()
                    status.empty()

                # ---------------------------------------------------------
                # Run the full PDF -> text -> LLM -> structured data pipeline
                # ---------------------------------------------------------
                with st.spinner("Reading PDF and extracting structured data..."):
                    outcome = run_extraction_pipeline(llm, uploaded_file, uploaded_file.name)

                # ---------------------------------------------------------
                # Persist results into session state so they survive reruns
                # ---------------------------------------------------------
                st.session_state["resume_text"] = outcome.resume_text
                st.session_state["num_pages"] = outcome.num_pages
                st.session_state["filename"] = outcome.filename
                st.session_state["structured_data"] = outcome.data
                st.session_state["error_message"] = None

                st.success(f"✅ Successfully extracted information from '{outcome.filename}'!")

            except (PDFReadError, ValueError) as known_err:
                # -----------------------------------------------------------
                # Expected/handled errors -> show a friendly message
                # -----------------------------------------------------------
                logger.error("Extraction failed: %s", known_err)
                st.session_state["error_message"] = str(known_err)
                st.error(f"❌ {known_err}")
            except Exception as unexpected_err:  # noqa: BLE001
                # -----------------------------------------------------------
                # Anything else -> log full traceback, show generic message
                # -----------------------------------------------------------
                logger.exception("Unexpected error during extraction.")
                st.session_state["error_message"] = str(unexpected_err)
                st.error(f"❌ An unexpected error occurred: {unexpected_err}")

    # --- Display previously extracted results (persists across reruns) ---
    if st.session_state["filename"]:
        # -------------------------------------------------------------
        # Show file name + page count banner if we have a prior result
        # -------------------------------------------------------------
        st.info(
            f"📎 File: **{st.session_state['filename']}** | "
            f"📄 Pages: **{st.session_state['num_pages']}**"
        )

    if st.session_state["resume_text"]:
        # -------------------------------------------------------------
        # Collapsible raw text viewer for the extracted PDF text
        # -------------------------------------------------------------
        with st.expander("📃 View Raw Extracted Text"):
            st.text_area("Raw Resume Text", st.session_state["resume_text"], height=300)

    if st.session_state["structured_data"]:
        # -------------------------------------------------------------
        # Render structured results + download buttons for last result
        # -------------------------------------------------------------
        st.divider()
        render_structured_results(st.session_state["structured_data"])
        st.divider()
        render_downloads(st.session_state["structured_data"], st.session_state["filename"])


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Entry point when run via `streamlit run app.py`
    # ---------------------------------------------------------------------
    main()
