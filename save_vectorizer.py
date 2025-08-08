import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Load your training data
df = pd.read_csv("data/processed/train_combined.csv")

# Use the appropriate column that contains the tweet text
text_column = "tweet"  # adjust if your column has a different name

# Fit vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(df[text_column])

# Save vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Vectorizer saved as models/vectorizer.pkl")
