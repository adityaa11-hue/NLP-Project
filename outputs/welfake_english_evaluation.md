# English article classifier: WELFake evaluation

Label mapping checked from the original WELFake release counts: label 0 is real (35,028 rows) and label 1 is fake (37,106 rows). The random test split is deduplicated but shares the benchmark's mixed-source construction; these results do not establish publisher/time-disjoint performance or truth on current news.

- Usable unique articles: 63,186
- Train+validation rows: 56,867
- Held-out test rows: 6,319
- Selected LogisticRegression C: 4
- Held-out accuracy: 0.9600
- Held-out macro-F1: 0.9596
- Confusion matrix (true rows genuine,misleading; predicted columns genuine,misleading): `[[3330, 122], [131, 2736]]`

```text
              precision    recall  f1-score   support

     genuine       0.96      0.96      0.96      3452
  misleading       0.96      0.95      0.96      2867

    accuracy                           0.96      6319
   macro avg       0.96      0.96      0.96      6319
weighted avg       0.96      0.96      0.96      6319

```

The application uses the article model for a binary screening label and retains evidence conflicts as Unable to Assess. The model score is not the probability that an article is true. WELFake was assembled from multiple older datasets and has source and topic limitations; claims require current evidence review.

Source record: https://zenodo.org/records/4561253. Original paper: https://doi.org/10.1109/TCSS.2021.3068519.
