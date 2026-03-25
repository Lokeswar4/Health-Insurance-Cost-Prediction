# Health Insurance Cost Prediction

Predicting medical insurance costs using patient demographics and health indicators. Built with scikit-learn on the [Kaggle Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance).

## Dataset

1,338 records with 7 attributes:

| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Patient age |
| sex | Categorical | Gender |
| bmi | Numeric | Body mass index |
| children | Numeric | Number of dependents |
| smoker | Categorical | Smoking status |
| region | Categorical | US residential region |
| charges | Numeric | Medical insurance cost (target) |

## Analysis Pipeline

1. **Exploratory Data Analysis** — distribution analysis, skewness/kurtosis, outlier detection (IQR method)
2. **Hypothesis Testing** — T-test and Mann-Whitney U test confirming smokers face significantly higher costs
3. **Feature Engineering** — One-hot encoding, dtype optimization, correlation-based feature selection
4. **Modeling** — Linear Regression with evaluation metrics (MAE, RMSE, MAPE, R-squared)

## Key Findings

- Smoking status is the strongest predictor of insurance costs
- Smokers pay on average 3-4x more than non-smokers (statistically significant, p < 0.001)
- BMI and age are secondary cost drivers

## Setup

```bash
pip install -r requirements.txt
jupyter notebook "Health Insurance Cost Prediction.ipynb"
```

## Future Improvements

- Add Random Forest and Gradient Boosting models for comparison
- Implement k-fold cross-validation (currently single train/test split)
- Add feature importance analysis
- Set random_state for reproducibility

## License

Apache 2.0
