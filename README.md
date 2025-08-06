📌 Problem Statement
Millions of women experience domestic violence, but many suffer in silence or use social media to express their distress subtly. Manually identifying these posts is not feasible due to the vast amount of online content. This project aims to develop a machine learning model to automatically detect self-reported domestic abuse in social media posts, enabling early intervention and support.

🎯 Objective
Build a text classification model that predicts whether a given social media post (e.g., tweet) is a self-report of domestic abuse or not. This can help support organizations, NGOs, or researchers quickly identify and respond to victims who may need help.

📥 Expected Model Inputs
The model takes as input:

A raw text post (e.g., tweet)

Example:
"I don't know what to do anymore. Every night he yells and throws things. I'm scared."

These posts are preprocessed with:

Lowercasing

Removal of punctuation, stopwords

Tokenization or embeddings (TF-IDF, Word2Vec, or BERT)

📤 Expected Model Output
A binary classification output:

1 → Post contains a report or indication of domestic abuse

0 → Post does not contain any abuse report

📈 Evaluation Metrics
Due to the potential class imbalance (fewer abuse-reporting posts than normal ones), we'll evaluate the model using:

F1-score (main metric) – balances precision and recall

Recall – important to capture all real abuse cases

Precision – ensures flagged posts are truly relevant

Accuracy – for general performance

ROC-AUC – for threshold-independent performance

✅ Success Criteria
The model should achieve an F1-score of at least 0.75 on the abuse-report class.

It should maintain a high recall, to ensure most abuse-reporting posts are captured.

Explainability: the model should highlight which keywords or phrases contributed to the prediction (e.g., via SHAP or attention visualization).

## Data Preprocessing Pipeline

The preprocessing pipeline includes the following steps:

1. **Handle Missing Values**: Impute missing values using mean/median for numerical features and mode for categorical features
2. **Encode Categorical Features**: Convert categorical variables to numerical using Label Encoding
3. **Normalize Numerical Features**: Standardize numerical features using StandardScaler
4. **Drop Irrelevant Columns**: Remove unnecessary columns from the dataset
5. **Create Features/Labels Split**: Separate features (X) from target variable (y)
6. **Train/Test Split**: Split data into training and testing sets using train_test_split()

### Usage
![MLOps CI/CD Pipeline](https://github.com/Efratsc/ML_fundamentals_proj/workflows/MLOps%20CI/CD%20Pipeline/badge.svg)

)

```python
from src.preprocess import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(target_column='your_target_column')

# Run complete pipeline
X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(df)
```

### Output Files

The preprocessing pipeline saves the following files in `data/processed/`:
- `X_train.csv`: Training features
- `X_test.csv`: Testing features  
- `y_train.csv`: Training labels
- `y_test.csv`: Testing labels
- `artifacts/scaler.pkl`: Fitted StandardScaler
- `artifacts/label_encoders.pkl`: Fitted LabelEncoders