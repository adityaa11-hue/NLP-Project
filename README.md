# Misinformation Shield

An English/Hindi Streamlit app for screening article text. Paste an article to surface factual claims, compare them with available fact-check records, and review one of three cautious outcomes: **Likely Genuine**, **Potentially Misleading**, or **Unable to Assess**.

This is a classroom/research prototype, not a source of definitive truth. Classifier confidence and evidence relevance are not probabilities that an article is true. A missing fact-check match does not prove a claim either way.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from this repository.
3. Choose branch `main` and app file `app.py`.
4. The app can run without a Google API key using its local X-FACT retrieval corpus. To enable live Google Fact Check searches, add `GOOGLE_FACT_CHECK_API_KEY = "your-key"` in the app's **Settings → Secrets**. Never commit the key to GitHub.

The deployment includes the trained English classifier and the X-FACT TSV corpus used for local evidence retrieval. The unrelated training CSVs, caches, experimental Hindi model, and browser-extension files are not part of this Streamlit deployment.

## Data and model notes

The English classifier uses the X-FACT claim dataset. Its source labels are mapped into the app's three screening outcomes; mixed ratings are treated as inconclusive. Hindi does not use the English classifier: it depends on retrieved evidence and may return **Unable to Assess** when no close fact-check match is available.

The project uses X-FACT, released under the MIT license. See the [dataset repository](https://github.com/utahnlp/x-fact), [dataset card](https://huggingface.co/datasets/utahnlp/x-fact), and [ACL paper](https://aclanthology.org/2021.acl-short.86/) for attribution and methodology.
