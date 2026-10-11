"""Upload a report and show what this boilerplate can actually extract."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import streamlit as st

from mediko.config import ConfigError, load_settings
from mediko.extraction.ocr import OcrError, PaddleOcrEngine
from mediko.extraction.validate import InputError
from mediko.pipeline import run_pipeline
from mediko.schema import RESEARCH_NOTICE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="MediKo", layout="wide")
st.title("MediKo")
st.warning(RESEARCH_NOTICE)

try:
    settings = load_settings()
except ConfigError as exc:
    st.error(str(exc))
    st.stop()


@st.cache_resource
def _engine(lang: str, device: str) -> PaddleOcrEngine:
    return PaddleOcrEngine(lang=lang, device=device)


def _show_result(result: dict) -> None:
    metadata = result["processing_metadata"]
    st.subheader(result["source_filename"])
    st.caption(f"Document {result['document_id']}")
    if metadata["ocr_used"]:
        pages = ", ".join(str(index + 1) for index in metadata["ocr_page_indexes"])
        st.info(
            f"PaddleOCR ran on pages {pages}. "
            f"Model: {metadata['model_version']}. {metadata['ocr_confidence']}"
        )
    else:
        st.info("PaddleOCR did not run. Embedded or plain text was enough for every page.")

    st.text_area("Extracted text", result["text"], height=280)

    for page in result["pages"]:
        with st.expander(_page_label(page), expanded=len(result["pages"]) == 1):
            if page["lines"]:
                st.dataframe(
                    [
                        {
                            "text": line["text"],
                            "model_confidence": line["model_confidence"],
                        }
                        for line in page["lines"]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
            elif page["text_source"] == "ocr":
                st.warning("PaddleOCR returned no text lines for this page.")
            st.text(page["text"] or "(empty)")

    st.subheader("Later pipeline stages")
    for name, component in result["components"].items():
        st.write(f"{name}: {component['status']}. {component['detail']}")

    st.download_button(
        "Download JSON",
        data=json.dumps(result, indent=2) + "\n",
        file_name="mediko-result.json",
        mime="application/json",
    )


def _page_label(page: dict) -> str:
    sources = {
        "embedded_text": "embedded text",
        "ocr": "PaddleOCR",
        "plain_text": "plain text",
    }
    source = sources.get(page["text_source"], page["text_source"])
    return f"Page {page['page_index'] + 1} · {source}"


uploaded = st.file_uploader("Medical report", type=["pdf", "txt"])
upload_name = uploaded.name if uploaded is not None else None
if st.session_state.get("source_name") != upload_name:
    st.session_state.source_name = upload_name
    st.session_state.result = None

process = st.button("Process report", type="primary", disabled=uploaded is None)

if process and uploaded is not None:
    if uploaded.size == 0:
        st.error("The file is empty.")
        st.session_state.result = None
    elif uploaded.size > settings.max_upload_bytes:
        st.error(
            f"The file is {uploaded.size} bytes, above the limit of "
            f"{settings.max_upload_bytes} bytes."
        )
        st.session_state.result = None
    else:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(uploaded.getvalue())
            temp_path = Path(handle.name)
        try:
            with st.spinner(
                "Extracting text. The first scanned page also downloads OCR models."
            ):
                st.session_state.result = run_pipeline(
                    temp_path,
                    source_filename=uploaded.name,
                    settings=settings,
                    engine=_engine(settings.ocr_lang, settings.ocr_device),
                )
        except (InputError, OcrError, ConfigError) as exc:
            st.session_state.result = None
            st.error(str(exc))
        except Exception as exc:
            st.session_state.result = None
            st.error(f"Processing failed: {exc}")
        finally:
            temp_path.unlink(missing_ok=True)

if st.session_state.get("result"):
    _show_result(st.session_state.result)
