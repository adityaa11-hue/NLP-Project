# English full-article model evaluation

The article-level model is trained on full article text from the fauxNeuz RealFakeNews educational corpus. This is a binary real/fake screening model, not a claim verifier. The application's X-FACT evidence retrieval remains an independent signal.

Dataset card: https://huggingface.co/datasets/fauxNeuz/RealFakeNews. The card lists CC BY-NC-SA 4.0, educational/non-commercial scope, and possible topic/source/label bias.

## Split and model selection

- English train rows: 9,595 (genuine=4,795, misleading=4,800).
- Exact normalized train/test duplicates excluded from training: 5.
- English test rows: 2,400; exact canonical duplicates within test: none.
- Candidate C was selected on a deterministic stratified 80/20 split of training data by validation macro-F1. Test labels were not used for candidate or threshold selection.
- Selected C=2; validation macro-F1=0.9181; validation selective accuracy=0.9226; validation coverage=0.9833; confidence cutoff=0.525.

## Held-out test

### Binary decision at 0.50

```text
Accuracy: 0.9092
Macro F1: 0.9090
              precision    recall  f1-score   support

     genuine       0.88      0.95      0.91      1200
  misleading       0.95      0.87      0.91      1200

    accuracy                           0.91      2400
   macro avg       0.91      0.91      0.91      2400
weighted avg       0.91      0.91      0.91      2400

Confusion matrix (true rows; genuine, misleading columns):
[[1140   60]
 [ 158 1042]]
```

### Three-way output with confidence cutoff 0.525

- Classified coverage: 0.9838 (2,361/2,400).
- Accuracy among classified articles: 0.9149.
- Unable to Assess abstentions: 39.

## Limitations

These metrics apply to this held-out benchmark split only. They do not establish accuracy across publishers, topics, dates, or current news. The source labels may be noisy, and the test set comes from the same dataset lineage as training. Treat the result as a model screening signal, not truth probability or verified fact-check. Independent publisher/time-disjoint evaluation is still needed.

## Post-evaluation data-quality audit

Manual inspection of the source dataset preview found wire-service reports (including Reuters stories) labeled `FAKE` and opinion/commentary pieces labeled `REAL`. The dataset card warns that labels may be inaccurate and says the dataset should not be used for automated fact-checking without human oversight. Therefore, despite the in-split score above, this artifact is **paused in the application by default**. Re-enable only for controlled experimentation with `MISINFO_ENABLE_UNVALIDATED_ENGLISH=1`; do not treat that as a recommended production setting. The English classifier should be retrained and evaluated on independently sourced, adjudicated labels before user-facing real/fake predictions resume.
