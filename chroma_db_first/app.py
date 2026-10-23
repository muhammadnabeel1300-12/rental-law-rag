import os
from pathlib import Path

import streamlit as st

from backend import (
    GeminiTemporaryError,
    generate_legal_answer,
    get_pdf_files,
    get_pdf_signature,
    load_collection,
    search_legal_documents,
)


@st.cache_resource(show_spinner=False)
def get_cached_collection(
    pdf_signature: tuple[tuple[str, int, int], ...],
):
    return load_collection(pdf_signature)


st.set_page_config(
    page_title="Pakistan Legal Information Assistant",
    page_icon="⚖️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            145deg, #f7f8f1 0%, #eef4ed 48%, #f6f3e9 100%
        );
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 1120px;
        padding-top: 2.4rem;
    }
    .eyebrow {
        color: #435e66;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .stApp h1 {
        color: #17324d;
        font-size: 2.55rem;
    }
    .intro {
        color: #556575;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    [data-testid="stSidebar"] { background: #e8edf1; }
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid #d5dfe5;
        border-radius: 8px;
        padding: 1.25rem;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid #d5dfe5;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

pdf_files = get_pdf_files()
pdf_signature = get_pdf_signature(pdf_files)

st.markdown(
    '<div class="eyebrow">Pakistan legal reference library</div>',
    unsafe_allow_html=True,
)
st.title("Pakistan Legal Information Assistant")
st.markdown(
    '<p class="intro">Ask a question about the laws in your PDFs. '
    'Answers use only retrieved legal text and include source details '
    'when available.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Library")
    st.write(f"**{len(pdf_files)}** PDFs available")
    st.caption("Place Pakistan legal PDFs in the pdfs folder.")
    result_count = st.slider(
        "Pages to retrieve",
        min_value=1,
        max_value=5,
        value=3,
    )
    gemini_api_key = st.text_input(
        "Gemini API key",
        type="password",
        value="",
    )

if not pdf_files:
    st.info(
        "Add a legal PDF to the pdfs folder, then refresh the app."
    )
    st.stop()

try:
    with st.spinner("Preparing semantic search..."):
        collection, page_count, chunk_count = get_cached_collection(
            pdf_signature
        )
except Exception as error:
    st.error(f"Could not load the document library: {error}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("PDF files", len(pdf_files))
col2.metric("Pages loaded", page_count)
col3.metric("Pages indexed", chunk_count)

with st.form("question_form"):
    question = st.text_input(
        "Your legal question",
        placeholder=(
            "Example: What does the law say about a tenant's notice period?"
        ),
    )
    submitted = st.form_submit_button(
        "Get answer from legal text",
        type="primary",
        width="stretch",
    )

if submitted:
    st.session_state.pop("answer", None)
    st.session_state.pop("search_results", None)
    st.session_state.pop("last_question", None)
    if not question.strip():
        st.warning("Please enter a question.")
    elif collection.count() == 0:
        st.warning("The document library is empty.")
    else:
        api_key = os.getenv("GEMINI_API_KEY") or gemini_api_key.strip()
        if not api_key:
            st.error(
                "Gemini is not configured. Enter your API key in the sidebar "
                "or set the GEMINI_API_KEY environment variable."
            )
        else:
            with st.spinner(
                "Searching legal documents and preparing an answer..."
            ):
                try:
                    search_results, context = search_legal_documents(
                        collection,
                        question.strip(),
                        result_count,
                    )

                    st.session_state["search_results"] = search_results
                    st.session_state["last_question"] = question.strip()

                    if not context:
                        st.session_state["answer"] = (
                            "The full paragraph was not retrieved."
                        )
                    else:
                        st.session_state["answer"] = generate_legal_answer(
                            question.strip(),
                            context,
                            api_key,
                        )
                except GeminiTemporaryError as error:
                    st.error(
                        "Gemini is temporarily unavailable. Your documents "
                        "were searched, but an answer could not be generated. "
                        "Please try again shortly. "
                        f"Details: {error}"
                    )
                except (RuntimeError, ValueError) as error:
                    st.error(f"Could not prepare a legal answer: {error}")
                except Exception as error:
                    st.error(f"Search failed: {error}")

if "search_results" in st.session_state:
    results = st.session_state["search_results"]

    st.subheader("Answer from retrieved legal text")
    st.caption(
        f"Question: {st.session_state['last_question']}"
    )

    if "answer" in st.session_state:
        st.text(st.session_state["answer"])

    st.subheader("Retrieved legal text")
    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    distances = results["distances"][0] or []

    if not documents:
        st.info("No relevant passages were found.")
    else:
        for index, (document, metadata, distance) in enumerate(
            zip(documents, metadatas, distances),
            start=1,
        ):
            source = Path(
                metadata.get("source", "Unknown law")
            ).name
            page = metadata.get("page", "Unknown")

            with st.expander(
                f"{index}. {source} · Page {page}",
                expanded=(index == 1),
            ):
                st.text(document)
                st.caption(
                    f"Cosine distance: {distance:.4f} "
                    "(lower generally means more similar)"
                )

    st.caption("AI answers may be incomplete; verify citations against the PDFs.")
