import html
import os

import streamlit as st

from misinfo.pipeline import analyze

st.set_page_config(page_title="Misinformation Shield", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

if not os.getenv("GOOGLE_FACT_CHECK_API_KEY"):
    try:
        key = st.secrets.get("GOOGLE_FACT_CHECK_API_KEY", "")
        if key:
            os.environ["GOOGLE_FACT_CHECK_API_KEY"] = key
    except Exception:
        pass

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {
  --ms-ink:#211638; --ms-muted:#514961; --ms-line:#e3ddef; --ms-paper:#f6f3fb;
  --ms-white:#fff; --ms-purple:#51299a; --ms-violet:#6e45bb; --ms-lavender:#f0eafd;
  --ms-green:#24734e; --ms-green-bg:#e8f5ed; --ms-red:#a83745; --ms-red-bg:#fbeaec;
  --ms-amber:#8a5a13; --ms-amber-bg:#fff4dd;
  --ms-space-1:clamp(6px, .65vw, 10px);
  --ms-space-2:clamp(10px, 1.1vw, 16px);
  --ms-space-3:clamp(14px, 1.7vw, 24px);
  --ms-space-4:clamp(20px, 2.6vw, 36px);
}
.stApp { background:var(--ms-paper); color:var(--ms-ink); font-family:'DM Sans','Segoe UI',sans-serif; }
.block-container { max-width:1240px; padding:clamp(16px, 2vw, 28px) clamp(14px, 2.5vw, 34px) clamp(36px, 4vw, 58px); }
header[data-testid="stHeader"] { background:transparent; }
.ms-topbar { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:15px 20px; margin:0 0 15px; color:#fff; border-radius:15px; background:linear-gradient(112deg,#3c1d79 0%,#51299a 62%,#6841b3 100%); box-shadow:0 8px 24px rgba(62,31,121,.16); }
.ms-brand { display:flex; align-items:center; gap:12px; min-width:0; }
.ms-brand-icon { display:grid; place-items:center; flex:0 0 42px; height:42px; border:1px solid rgba(255,255,255,.28); border-radius:13px; background:rgba(255,255,255,.12); font-size:23px; }
.ms-title { color:#fff; font:800 19px/1.2 Manrope,'Segoe UI',sans-serif; letter-spacing:-.25px; }
.ms-subtitle { margin-top:3px; color:rgba(255,255,255,.82); font-size:12px; }
.ms-live { display:flex; align-items:center; gap:8px; padding:7px 11px; border:1px solid rgba(255,255,255,.24); border-radius:999px; color:#f7f2ff; font-size:11px; font-weight:600; white-space:nowrap; }
.ms-dot { width:7px; height:7px; border-radius:50%; background:#9be2bc; box-shadow:0 0 0 3px rgba(155,226,188,.14); }
.ms-eyebrow { color:var(--ms-violet); font-size:10px; font-weight:700; letter-spacing:1.25px; text-transform:uppercase; }
.ms-panel-title { margin:3px 0 5px; color:var(--ms-ink); font:700 17px/1.3 Manrope,'Segoe UI',sans-serif; }
.ms-panel-copy { color:var(--ms-muted); font-size:12px; line-height:1.55; }
.ms-guide-row { padding:11px 0; border-bottom:1px solid #eee9f5; }
.ms-guide-row:last-child { border-bottom:0; padding-bottom:1px; }
.ms-label { display:inline-flex; align-items:center; gap:7px; margin:0 0 5px; padding:5px 9px; border-radius:999px; font-size:11px; font-weight:700; }
.ms-label-genuine { color:var(--ms-green); background:var(--ms-green-bg); }
.ms-label-misleading { color:var(--ms-red); background:var(--ms-red-bg); }
.ms-label-unable { color:var(--ms-amber); background:var(--ms-amber-bg); }
.ms-note { margin-top:12px; padding:11px 12px; border:1px solid #e8def8; border-radius:10px; color:#52436c; background:#f7f3ff; font-size:11px; line-height:1.5; }
.ms-result { display:flex; align-items:flex-start; gap:14px; padding:20px 22px; border:1px solid var(--ms-line); border-left:5px solid #8a8297; border-radius:14px; background:#fff; box-shadow:0 5px 17px rgba(42,26,68,.045); }
.ms-result-genuine { border-left-color:var(--ms-green); }
.ms-result-misleading { border-left-color:var(--ms-red); }
.ms-result-unable { border-left-color:#cb902a; }
.ms-result-mark { display:grid; place-items:center; flex:0 0 38px; width:38px; height:38px; border-radius:12px; color:var(--ms-purple); background:var(--ms-lavender); font-size:18px; }
.ms-result-label { margin:0 0 5px; color:var(--ms-ink); font:800 22px/1.25 Manrope,'Segoe UI',sans-serif; letter-spacing:-.3px; }
.ms-result-text { color:#5c536b; font-size:13px; line-height:1.55; }
.ms-source-card { margin:9px 0 5px; padding:14px 15px; border:1px solid var(--ms-line); border-radius:12px; background:linear-gradient(180deg,#fff,#fdfcff); }
.ms-source-top { display:flex; flex-wrap:wrap; justify-content:space-between; gap:6px 12px; margin-bottom:7px; color:#736b81; font-size:11px; }
.ms-source-rating { display:inline-block; margin-bottom:6px; padding:4px 8px; border-radius:999px; color:#49317a; background:#f0eafd; font-size:11px; font-weight:700; }
.ms-source-title { color:var(--ms-ink); font-size:13px; font-weight:700; line-height:1.45; }
.ms-source-excerpt { margin-top:8px; color:#6d6680; font-size:12px; line-height:1.5; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"] { color:var(--ms-ink); }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p { color:#514961 !important; font-size:13px; line-height:1.55; }
.stApp [data-testid="stMetric"] { display:flex; min-width:0; min-height:124px; padding:18px 20px; align-items:center; }
.stApp [data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width:0; }
.stApp [data-testid="stMetricLabel"] { margin-bottom:7px; color:#514961 !important; font-size:13px; font-weight:650; }
.stApp [data-testid="stMetricValue"] { min-width:0; color:#211638 !important; font-size:clamp(16px,1.45vw,22px) !important; line-height:1.25; white-space:normal !important; overflow-wrap:anywhere; }
.stApp [data-testid="stMetricValue"] *, .stApp [data-testid="stMetricValue"] > div, .stApp [data-testid="stMetricValue"] p { max-width:100%; white-space:normal !important; overflow-wrap:anywhere !important; word-break:break-word; }
.stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stWidgetLabel"] p { color:#302544 !important; font-weight:700; }
.stButton>button[kind="primary"] { min-height:46px; border:1px solid var(--ms-purple); border-radius:10px; color:#fff; background:linear-gradient(100deg,#51299a,#6940b0); box-shadow:0 5px 13px rgba(81,41,154,.15); font-weight:700; transition:transform .14s ease,box-shadow .14s ease,background .14s ease; }
.stButton>button[kind="primary"]:hover { border-color:#43207f; background:#43207f; box-shadow:0 7px 16px rgba(81,41,154,.21); transform:translateY(-1px); }
.stButton>button { border-radius:9px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--ms-line) !important; border-radius:14px !important; background:#fff; box-shadow:0 4px 15px rgba(42,26,68,.035); }
[data-testid="stTabs"] { margin-top:var(--ms-space-2); }
[data-testid="stTabs"] [role="tablist"] { gap:var(--ms-space-1); padding:var(--ms-space-1) var(--ms-space-2); border:1px solid var(--ms-line); border-radius:12px; background:#eee9f6; }
[data-testid="stTabs"] button[role="tab"] { min-height:37px; border-radius:8px; color:#433950 !important; opacity:1 !important; font-size:13px; font-weight:700; }
[data-testid="stTabs"] button[role="tab"] *, [data-testid="stTabs"] button[role="tab"] [data-testid="stMarkdownContainer"] p { color:inherit !important; opacity:1 !important; }
[data-testid="stTabs"] button[role="tab"]:hover { color:#321b59 !important; background:#f8f5fd; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#3e1e73 !important; background:#fff; box-shadow:0 2px 6px rgba(43,27,69,.08); }
[data-testid="stTabs"] [data-testid="stHorizontalBlock"] { gap:clamp(12px,2.2vw,30px); }
[data-testid="stTabs"] [data-testid="stVerticalBlock"] { gap:clamp(8px,1.2vw,16px); }
[data-testid="stMetric"] { padding:12px 14px; border:1px solid var(--ms-line); border-radius:12px; background:#fff; }
[data-testid="stMetricLabel"] { color:var(--ms-muted); font-size:11px; }
[data-testid="stMetricValue"] { color:var(--ms-ink); font-family:Manrope,'Segoe UI',sans-serif; font-size:20px; }
[data-testid="stAlert"] p, [data-testid="stAlert"] [data-testid="stMarkdownContainer"], [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p { color:#fff !important; }
div[data-testid="stTextArea"] textarea { min-height:210px; border:1px solid #ded5ed; border-radius:11px; background:#fdfcff; color:var(--ms-ink); line-height:1.6; }
div[data-testid="stTextArea"] textarea::placeholder { color:#625970 !important; opacity:1 !important; }
div[data-testid="stTextArea"] textarea:focus { border-color:#7650bd; box-shadow:0 0 0 3px rgba(110,69,187,.12); }
[data-testid="stRadio"] { margin:var(--ms-space-1) 0 var(--ms-space-2); }
[data-testid="stRadio"] > div { gap:8px; }
[data-testid="stRadio"] label { padding:7px 12px; border:1px solid #d8cfe6; border-radius:999px; background:#fff; color:#302544 !important; opacity:1 !important; font-weight:650; }
[data-testid="stRadio"] label *, [data-testid="stRadio"] label p { color:#302544 !important; opacity:1 !important; }
[data-testid="stRadio"] label:has(input:checked) { border-color:#7049b7; background:#f0eafd; }
input[type="radio"] { accent-color:#51299a; }
a { color:#58329c !important; font-weight:600; }
@media(max-width:760px) {
  .block-container { padding: .85rem .8rem 2rem; }
  .ms-topbar { padding:12px 13px; border-radius:12px; }
  .ms-brand-icon { flex-basis:36px; height:36px; font-size:19px; }
  .ms-title { font-size:16px; }
  .ms-subtitle { max-width:225px; font-size:10px; }
  .ms-live { padding:6px 8px; font-size:10px; }
  .ms-result { padding:16px; }
  .ms-result-label { font-size:19px; }
  [data-testid="stTabs"] [role="tablist"] { flex-wrap:wrap; }
}
</style>
<div class="ms-topbar">
  <div class="ms-brand"><div class="ms-brand-icon">🛡</div><div><div class="ms-title">Misinformation Shield</div><div class="ms-subtitle">Multilingual claim screening · English &amp; Hindi</div></div></div>
  <div class="ms-live"><span class="ms-dot"></span> Evidence-aware analysis</div>
</div>
""", unsafe_allow_html=True)
st.caption("A research screening aid. Model scores are not truth probabilities, and related fact-checks need human review.")


def _render_source(match: dict) -> None:
    title = match.get("title") or match.get("claim") or "Fact-check record"
    publisher = match.get("publisher") or match.get("provider") or "Publisher unavailable"
    date = match.get("review_date") or "Date not provided"
    score = match.get("similarity")
    rating = match.get("rating") or "Rating not provided"
    relevance = f"{score:.0%}" if isinstance(score, (int, float)) else "Not available"
    escaped_publisher = html.escape(str(publisher))
    escaped_date = html.escape(str(date))
    escaped_rating = html.escape(str(rating))
    escaped_title = html.escape(str(title))
    escaped_excerpt = html.escape(str(match.get("excerpt", "")))
    excerpt_html = f'<div class="ms-source-excerpt">{escaped_excerpt}</div>' if escaped_excerpt else ""
    st.markdown(
        f"<div class='ms-source-card'><div class='ms-source-top'><span>{escaped_publisher}</span><span>{escaped_date}</span></div>"
        f"<span class='ms-source-rating'>{escaped_rating} · Relevance {relevance}</span>"
        f"<div class='ms-source-title'>{escaped_title}</div>"
        f"{excerpt_html}</div>",
        unsafe_allow_html=True,
    )
    if match.get("url"):
        st.markdown(f"[Open original fact-check ↗]({match['url']})")
    st.caption("Relevance is text similarity only; it is not a truth probability.")


with st.container(border=True):
    input_col, guide_col = st.columns([1.55, 0.85], gap="large")
    with input_col:
        st.markdown('<div class="ms-eyebrow">Start a review</div><div class="ms-panel-title">Analyze an article</div>', unsafe_allow_html=True)
        st.markdown('<div class="ms-panel-copy">Paste article text to surface factual claims and compare them with available fact-check records.</div>', unsafe_allow_html=True)
        language = st.radio("Article language", ["English", "Hindi"], horizontal=True)
        text = st.text_area("Article text", height=225, placeholder="Paste an article containing factual statements you want to review…", help="Paste only the article text. Avoid adding labels such as ‘fake’, ‘true’, or ‘training example’; those words can change classifier predictions. The system selects a small number of factual-looking sentences for screening and source search.")
        analyze_clicked = st.button("🛡  Analyze article", type="primary", use_container_width=True)
    with guide_col:
        st.markdown('<div class="ms-eyebrow">Understand the result</div><div class="ms-panel-title">Three possible outcomes</div>', unsafe_allow_html=True)
        st.markdown('<div class="ms-guide-row"><span class="ms-label ms-label-genuine">● Likely Genuine</span><div class="ms-panel-copy">The model or close supporting fact-checks lean toward accurate reporting.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="ms-guide-row"><span class="ms-label ms-label-misleading">● Potentially Misleading</span><div class="ms-panel-copy">A claim resembles misleading examples or has close negative fact-check evidence.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="ms-guide-row"><span class="ms-label ms-label-unable">● Unable to Assess</span><div class="ms-panel-copy">Evidence is missing, uncertain, or conflicts with the model.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="ms-note"><b>Keep in mind</b><br>Open the original sources. A similar claim may have a different date, wording, or context.</div>', unsafe_allow_html=True)

if analyze_clicked:
    with st.spinner("Extracting claims and searching for related fact-checks…"):
        result = analyze(text, "hi" if language == "Hindi" else "en")
    st.markdown("### Review result")
    kind = {"Likely Genuine": "genuine", "Potentially Misleading": "misleading", "Unable to Assess": "unable"}[result["label"]]
    st.markdown(
        f"<div class='ms-result ms-result-{kind}'><div class='ms-result-mark'>✦</div><div><div class='ms-result-label'>{html.escape(result['label'])}</div><div class='ms-result-text'>{html.escape(result['explanation'])}</div></div></div>",
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("Classifier prediction", result.get("classifier_label", "Unavailable"))
    score = result.get("classifier_score")
    metric_cols[1].metric("Classifier score", f"{score:.0%}" if score is not None else "Not available")
    metric_cols[2].metric("Retrieved matches", result.get("evidence_summary", {}).get("matching_records", 0))
    if result.get("model_scope") == "whole_article":
        st.caption("The English classifier scores the complete article. Its score is model confidence, not the probability that the article is true. Retrieved fact-check evidence is shown separately.")
    elif result.get("model_status") == "paused_noisy_training_labels":
        st.caption("The English classifier is paused because its training labels failed a data-quality review. No result is inferred from an unreliable model score.")
    else:
        st.caption("Classifier score describes model confidence between learned classes. Evidence relevance describes text similarity. Neither is the probability a claim is true.")

    summary_tab, analysis_tab, claims_tab, sources_tab, method_tab = st.tabs(["Summary", "Detailed Analysis", "All Claims", "Sources", "Methodology"])
    with summary_tab:
        summary_left, summary_right = st.columns([1.1, 0.9], gap="large")
        with summary_left:
            st.markdown("#### At a glance")
            st.write(result["evidence_status"])
            st.write(f"**Classifier leaning:** {result.get('classifier_label', 'Unavailable')}")
            st.write(f"**Records retrieved:** {result.get('evidence_summary', {}).get('matching_records', 0)}")
            extraction = result.get("extraction_summary", {})
            if extraction:
                st.write(f"**Factual-looking statements selected:** {extraction.get('selected_claims', 0)}")
                excluded = extraction.get("excluded_statements", [])
                if excluded:
                    with st.expander(f"Statements not assessed ({len(excluded)})"):
                        for item in excluded:
                            st.write(f"- {item.get('text', '')} — {item.get('reason', 'not selected')}")
                st.caption(extraction.get("method", ""))
        with summary_right:
            if language == "Hindi" and not result.get("model_available"):
                st.info("Hindi classification is unavailable because the current training data does not support a reliable model. This assessment relies on retrieved fact-check evidence and may remain Unable to Assess.")
            elif language == "Hindi":
                st.warning("The Hindi whole-article classifier is experimental. Its benchmark separates real-news and fake-news folders, which may also separate publishers or writing styles. The score is not a truth probability; review the article’s original sources and any conflicting evidence.")
            elif result.get("model_status") == "paused_noisy_training_labels":
                st.warning("English model predictions are paused: the training corpus contains mislabeled news examples. The app will use strong fact-check matches when available and otherwise return Unable to Assess, rather than present an unreliable real/fake label.")
            elif not result.get("model_available"):
                st.warning("The English model could not be loaded. The assessment uses available retrieval evidence only.")
            if any(claim.get("evidence", {}).get("google_status") == "not_configured" for claim in result.get("claims", [])):
                st.info("Live Google Fact Check search is not configured for this app. Add GOOGLE_FACT_CHECK_API_KEY to the app's environment or Streamlit secrets to search published fact-checks; without a close match, the result can remain Unable to Assess.")
            if result.get("evidence_summary", {}).get("search_errors"):
                st.error("Some evidence searches failed. Review the API/network details in Detailed Analysis.")

    with analysis_tab:
        st.markdown("#### Model signal and evidence signal")
        if result.get("model_scope") == "whole_article":
            model_language = "Hindi" if language == "Hindi" else "English"
            st.markdown(f'<div class="ms-panel-copy">The {model_language} classifier predicts from the complete article. Each extracted claim is shown separately for fact-check evidence; retrieved ratings can conflict with the article-level screening signal.</div>', unsafe_allow_html=True)
        elif result.get("model_status") == "paused_noisy_training_labels":
            st.warning("English classifier paused: the training corpus contains mislabeled examples. Claims below are checked against fact-check records only; an empty search is not evidence for either side.")
        else:
            st.markdown('<div class="ms-panel-copy">The classifier prediction and published fact-check ratings remain separate. Very close, interpretable evidence can support or conflict with a model result; conflicting signals lead to Unable to Assess.</div>', unsafe_allow_html=True)
        for index, claim in enumerate(result.get("claims", []), 1):
            with st.container(border=True):
                st.markdown(f"**Claim {index}** · {claim['label']}")
                st.write(claim["claim"])
                detail_cols = st.columns(3)
                detail_cols[0].caption(f"Classifier: {claim.get('classifier_label', 'Unavailable')}")
                detail_cols[1].caption(f"Evidence: {claim.get('evidence_assessment', {}).get('verdict', 'insufficient')}")
                detail_cols[2].caption(f"Close matches: {claim.get('evidence_assessment', {}).get('strong_match_count', 0)}")
                metadata = claim.get("extraction", {}).get("metadata", {})
                metadata_items = [(label, metadata.get(key, [])) for label, key in (("Entities", "entities"), ("Dates", "dates"), ("Quantities", "quantities")) if metadata.get(key)]
                if metadata_items:
                    st.caption(" · ".join(f"{label}: {', '.join(values)}" for label, values in metadata_items))
                st.caption(claim.get("decision_reason", ""))
                errors = []
                local_error = claim.get("evidence", {}).get("local_error")
                google_error = claim.get("evidence", {}).get("google_error") if claim.get("evidence", {}).get("google_status") == "error" else None
                if local_error:
                    errors.append(local_error)
                if google_error:
                    errors.append(google_error)
                if errors:
                    st.error(" ".join(errors))
                elif claim.get("evidence", {}).get("google_status") == "not_configured":
                    st.caption("Live Google search is off; the local X-FACT index was searched.")
                elif claim.get("evidence", {}).get("status") == "no_results":
                    st.info("No sufficiently relevant fact-check record was found. No match does not support either side.")

    with claims_tab:
        claims = result.get("claims", [])
        if result.get("model_status") == "paused_noisy_training_labels":
            st.warning("No claim-level classifier score is shown because the English classifier is paused after a training-label quality review. To search published fact-checks live, configure GOOGLE_FACT_CHECK_API_KEY in the app's Streamlit secrets.")
        if not claims:
            st.info(result["explanation"])
        for index, claim in enumerate(claims, 1):
            with st.container(border=True):
                claim_head, claim_status = st.columns([1.4, 0.6])
                claim_head.markdown(f"#### Claim {index}")
                claim_status.markdown(f"**{claim['label']}**")
                st.write(f"> {claim['claim']}")
                st.caption(f"Classifier: {claim.get('classifier_label', 'Unavailable')}")
                if claim.get("classifier_score") is not None:
                    st.caption(f"Classifier score: {claim['classifier_score']:.0%} · not a truth probability")
                elif result.get("model_scope") == "whole_article":
                    st.caption("This claim is compared with fact-check evidence separately; the classifier score applies to the full article above.")
                elif result.get("model_status") == "paused_noisy_training_labels":
                    st.caption("Classifier score intentionally withheld: training labels were found to be unreliable.")
                else:
                    st.caption("No reliable claim-level classifier score is available.")
                metadata = claim.get("extraction", {}).get("metadata", {})
                metadata_items = [(label, metadata.get(key, [])) for label, key in (("Entities", "entities"), ("Dates", "dates"), ("Quantities", "quantities")) if metadata.get(key)]
                if metadata_items:
                    st.caption(" · ".join(f"{label}: {', '.join(values)}" for label, values in metadata_items))
                matches = claim.get("evidence", {}).get("matches", [])
                if matches:
                    assessments = claim.get("evidence_assessment", {}).get("match_assessments", [])
                    for match_index, match in enumerate(matches[:3]):
                        _render_source(match)
                        if match_index < len(assessments):
                            assessed = assessments[match_index]
                            st.caption(f"Evidence role: {assessed.get('relevance', 'related_inconclusive')} · published rating direction: {assessed.get('published_rating_direction', 'unknown')}")
                else:
                    st.info(claim.get("evidence", {}).get("message", "No sufficiently relevant fact-check record was found for this claim."))

    with sources_tab:
        matches = result.get("sources", [])
        if not matches:
            st.info("No sufficiently relevant fact-check record was found for the extracted claims. This does not support either side of the claim.")
        for match in matches:
            _render_source(match)

    with method_tab:
        st.markdown("#### How analysis works")
        st.markdown("1. The English article model analyzes the complete pasted article using word 1–2 gram hashing features and Logistic Regression; the training script selects model regularization on validation data and applies a 0.5 binary decision threshold.  \n2. A rule-based screen separately selects factual-looking sentences and lists opinions/commentary or other skipped statements; extracted entities, dates, and quantities are literal text cues, not trained NER.  \n3. Search X-FACT against those candidate claims using text similarity, entity overlap, and quantity agreement. Optional Google Fact Check Tools API results are ranked the same way.  \n4. A search match is only related unless its wording/context is sufficiently close. Fact-check ratings stay separate from the article model; strong contradictions or conflicting ratings lead to Unable to Assess.")
        st.markdown("**English classifier:** the complete article is screened by a Logistic Regression model trained on the original WELFake benchmark (label 0 mapped to real, label 1 to fake after checking the original class counts). Its held-out score measures performance on a random split of this same mixed-source benchmark, not verified truth or generalization to current publishers. The score is not a probability that an article is true. Strong fact-check evidence is compared separately, and direct conflicts may still yield Unable to Assess.\n\n**Hindi classifier:** the complete article is screened by an experimental character n-gram model trained on the public Hindi real/fake corpus (Zenodo record 11408513). Its random split can retain publisher, source and writing-style differences between labels, so its near-99% in-corpus result must not be read as real-world accuracy. Strong conflicting evidence can still produce Unable to Assess.")
