# Hindi baseline evaluation

Source: Zenodo record 11408513 (CC BY 4.0); 2400 exact-deduplicated text files.
Split sizes: train 1680, validation 360, test 360.
Validation-selected thresholds on P(misleading): genuine <= 0.500; misleading >= 0.525; otherwise abstain.
Validation fake precision at selected threshold: 1.0.
Held-out fake precision: 1.0000; coverage: 0.9972.

## Held-out test report

```text
                        precision    recall  f1-score   support

        Likely Genuine       0.99      1.00      0.99       180
Potentially Misleading       1.00      0.98      0.99       180
      Unable to Assess       0.00      0.00      0.00         0

              accuracy                           0.99       360
             macro avg       0.66      0.66      0.66       360
          weighted avg       0.99      0.99      0.99       360

```

## Limitations

This dataset is organized into real-news and fake-news folders and the sources may differ by class. Files are randomly split after exact deduplication; source and near-duplicate leakage may remain. The report is an in-dataset baseline, not a real-world accuracy estimate. The model must be shown as experimental and uses a validation-selected abstain band.