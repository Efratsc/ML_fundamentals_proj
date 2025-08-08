🧠 Domestic Abuse Detection from Tweets – MLOps Pipeline
📌 Problem Statement
Millions of women experience domestic violence, but many suffer in silence or use social media to express their distress subtly. Manually identifying these posts is not feasible due to the vast amount of online content.
This project develops a machine learning model to automatically detect self-reported domestic abuse in social media posts, enabling early intervention and support.

🎯 Objective
Build a text classification model that predicts whether a given tweet is a self-report of domestic abuse.
This model can help support organizations, NGOs, or researchers quickly identify and respond to victims who may need help.

📥 Model Input
The model takes as input:

A raw tweet (text only)

Example:

"I don't know what to do anymore. Every night he yells and throws things. I'm scared."

Preprocessing steps:

Lowercasing

Removal of punctuation and stopwords

Tokenization using TF-IDF

📤 Model Output
A binary classification result:

1 → Tweet contains an indication of domestic abuse

0 → Tweet does not contain any abuse report

📈 Evaluation Metrics
F1-score (primary) – balances precision and recall

Recall – capture all real abuse cases

Precision – minimize false positives

Accuracy – overall performance

ROC-AUC – threshold-independent performance

✅ Success Criteria
F1-score of at least 0.75 on abuse-reporting class

High recall to avoid missing any real abuse indicators

(Optional) Support model explainability using SHAP or keyword importance

🧪 Data Preprocessing Pipeline
The pipeline performs:

Lowercasing & Cleaning: Text normalization

TF-IDF Vectorization: Transform text into features

Train/Test Split: Prepare datasets for training and evaluation

Label Mapping: Map the class column to binary format (if needed)

Output Files
X_train.csv, X_test.csv, y_train.csv, y_test.csv → under data/processed/

models/model.pkl → trained classifier

models/vectorizer.pkl → saved TF-IDF vectorizer

🚀 Deployment
The API is deployed using FastAPI and Uvicorn via Docker.

To run locally:

bash
Copy
Edit
docker build -t domestic-abuse-detector .
docker run -p 8000:8000 domestic-abuse-detector
Visit: http://localhost:8000/docs

🔁 CI/CD
This project uses GitHub Actions for continuous integration and deployment:



🧪 Example Usage
bash
Copy
Edit
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"tweet_text": "He hit me again last night. I’m afraid to go home."}'
Response:

json
Copy
Edit
{
  "prediction": 1,
  "label": "Abusive"
}