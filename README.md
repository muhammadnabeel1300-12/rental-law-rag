# rental-law-rag

A Streamlit legal-document RAG app that retrieves relevant passages from
Pakistan legal PDFs and reports their source and page details.

## Run locally

From the repository root, install the app dependencies and launch Streamlit:

```shell
python -m pip install -r chroma_db_first/requirements.txt
python -m streamlit run chroma_db_first/app.py
```

`app.py` contains the Streamlit interface. `backend.py` contains PDF indexing,
ChromaDB retrieval, context preparation, and Gemini answer generation.

The app expects PDF files in `chroma_db_first/pdfs/`. That folder and the
generated ChromaDB database are intentionally excluded from Git. Add only
documents you are permitted to publish, or arrange another approved way to
make the documents available to the deployed app.

The Gemini key can be entered in the app's password field or provided through
the `GEMINI_API_KEY` environment variable. Do not commit API keys or other
secrets.

## Deploy on Streamlit Community Cloud

This project is a Streamlit app, not a Python HTTP function. Deploy it on
[Streamlit Community Cloud](https://share.streamlit.io/):

1. Select **Create app** and choose **Yup, I have an app**.
2. Set the repository to `muhammadnabeel1300-12/rental-law-rag`.
3. Set the branch to `main`.
4. Set the main file path to `chroma_db_first/app.py`.
5. Deploy the app. Community Cloud finds the dependencies in
   `chroma_db_first/requirements.txt`.
6. Enter the Gemini API key in the app's password field.

The app will need its legal PDFs available in the deployment environment
before it can answer questions. The repository deliberately does not include
the local PDFs or ChromaDB files.

## Vercel compatibility

Vercel's Python runtime loads ASGI/WSGI HTTP applications such as FastAPI or
Flask from a top-level `app`, `application`, or `handler` variable. `app.py`
instead runs Streamlit UI commands and does not expose an HTTP application.
Adding a placeholder variable or a `vercel.json` rewrite does not run the
Streamlit server, so this app cannot be deployed directly as a Vercel Python
Function.

To keep the existing interface and RAG pipeline, use Streamlit Community
Cloud or another host that supports running Streamlit. Using Vercel would
require a separate web-application architecture and is not a configuration
change to this Streamlit script.
