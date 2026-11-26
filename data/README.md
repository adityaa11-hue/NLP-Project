# Data sources and preparation

## X-FACT

The corpus is the MIT-licensed [X-FACT dataset](https://github.com/utahnlp/x-fact), with an [ACL 2021 paper](https://aclanthology.org/2021.acl-short.86/) and [dataset card](https://huggingface.co/datasets/utahnlp/x-fact). The project uses fact-checker-labeled claim statements and source evidence fields; cite the paper in project submissions.

Download with `python scripts/download_datasets.py` and prepare with `python scripts/prepare_datasets.py`. The shared parser handles rows containing unescaped tabs by realigning the trailing dates, claimant, claim, and rating. Unalignable records are skipped rather than assigned a guessed label.

Raw X-FACT files are stored under `x-fact/`; generated `claim_*.csv` splits are stored in this directory. Both raw data and generated splits are excluded from Git.

## Label mapping and splits

| Original rating | App label | Binary model target |
|---|---|---|
| true, mostly true | Likely Genuine | genuine |
| false, mostly false, pants on fire | Potentially Misleading | misleading |
| half true, partly true/misleading, other, barely true | Unable to Assess | excluded from binary training; used to validate abstention |

English records come from PolitiFact in this release. Normalized exact duplicates are removed before a chronological train/validation/test split. All English results therefore measure later-period performance from the same publisher; cross-publisher generalization is unknown.

The complete-article English screening model is trained separately on the original [WELFake Zenodo release](https://zenodo.org/records/4561253), described in the [original paper](https://doi.org/10.1109/TCSS.2021.3068519). The script checks the source class counts (35,028 label-0 real articles and 37,106 label-1 fake articles), removes exact normalized duplicates before a stratified split, chooses Logistic Regression regularization on validation data, and reports a held-out random-split evaluation. Rebuild the local artifact with `python scripts/train_welfake_english.py`; the script downloads the raw CSV into `data/welfake/`, which is ignored by Git. The committed model artifact does not require the raw 245 MB dataset at app runtime. See `outputs/welfake_english_evaluation.md` for the measured score and limitations. The random split is not publisher- or time-disjoint, so its accuracy is not a guarantee for current articles.

Hindi X-FACT train records include Potentially Misleading and Unable to Assess but no Likely Genuine examples. Hindi in-domain and OOD release splits are preserved for the audit, but they cannot support a reliable three-way Hindi classifier. The old Hindi workbook has 6,157 “authentic” and 6,487 “fake” rows, but the fake class consists of fact-check writeups while the authentic class consists of ordinary news. That source/style confounding makes it unsuitable as a binary article classifier here.

The Hindi app model is the experimental character n-gram Logistic Regression baseline trained by `python scripts/train_hindi.py` from Zenodo record 11408513 (CC BY 4.0). It is now loaded by default for Hindi whole-article screening. Its saved decision band applies to the misleading-class probability. The underlying dataset uses separate real-news and fake-news folders and may encode publisher/style cues; its random-split score is not a reliable estimate of performance on new Hindi publishers or current news. The 59 MB archive is ignored by Git; the small trained artifact `models/hindi_misinfo_experimental.joblib` is included so deployment does not require retraining.

The [HinFakeNews listing](https://aikosh.indiaai.gov.in/home/datasets/details/hinfakenews_2.html) describes a larger Hindi corpus but likewise separates ordinary articles from fact-checker-sourced fake examples, so it does not resolve this project’s source confounding. [IndicClaimBuster](https://github.com/pritampal98/indic-claim-buster) is a relevant multilingual claim-verification resource, but its dataset is password-protected; its paper also documents LLM-generated refutations, which would require explicit treatment in any later training.

## Retrieval

Local search compares submitted claims against X-FACT claims using word and character n-gram TF-IDF plus conservative entity-like span and numeric agreement adjustments. Fact-check ratings, source URLs, review dates, and evidence excerpts are shown as records, not proof. Optional live search requires `GOOGLE_FACT_CHECK_API_KEY`. A successful empty search is distinct from API/network failure; no match is not evidence for either side.
