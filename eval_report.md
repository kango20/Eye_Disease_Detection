# Model Evaluation Report

## Dataset Information

- **Train Set Size:** 3795 images belonging to 4 classes.
- **Validation Set Size:** 211 images belonging to 4 classes.
- **Test Set Size:** 211 images belonging to 4 classes.
- 
## Metric Descriptions

- **Accuracy**: The proportion of true results among the total number of cases examined. It measures how often the model is correct across all classes.

- **Precision**: The ratio of true positive predictions to the total number of positive predictions. It evaluates the model's ability to identify only relevant instances.

- **Recall (Sensitivity)**: The ratio of true positive predictions to the total number of actual positives. It assesses the model's ability to detect all relevant instances.

- **F1 Score**: The harmonic mean of precision and recall. It balances both metrics, especially useful when their values vary significantly.

- **ROC AUC Score**: The Area Under the Receiver Operating Characteristic Curve. It measures the model's ability to discriminate between classes, with a higher score indicating better performance.

## Training Performance

- **Loss:** 0.008962206542491913
- **Accuracy:** 99.71%
- **Precision:** 99.71%
- **Recall:** 99.71%
- **F1 Score:** 99.71%
- **ROC AUC Score:** 99.998%

## Validation Performance

- **Loss:** 0.4641197919845581
- **Accuracy:** 93.84%
- **Precision:** 93.84%
- **Recall:** 93.84%
- **F1 Score:** 94.20%
- **ROC AUC Score:** 97.49%

## Test Performance

- **Loss:** 0.2605193555355072
- **Accuracy:** 94.79%
- **Precision:** 94.79%
- **Recall:** 94.79%
- **F1 Score:** 94.78%
- **ROC AUC Score:** 98.38%

## Classification Report

| Class                | Precision | Recall | F1-Score | Support |
|----------------------|-----------|--------|----------|---------|
| Cataract             | 0.94      | 0.96   | 0.95     | 50      |
| Diabetic Retinopathy | 1.00      | 1.00   | 1.00     | 54      |
| Glaucoma             | 0.93      | 0.88   | 0.90     | 48      |
| Normal               | 0.92      | 0.95   | 0.93     | 59      |

- **Accuracy:** 95%
- **Macro Avg:** 95%
- **Weighted Avg:** 95%
