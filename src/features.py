import os
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy import sparse

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_features(input_file: str, output_dir: str):
    logging.info(f"Loading cleaned data from {input_file}")
    df = pd.read_csv(input_file)

    if 'tweet' not in df.columns or 'class' not in df.columns:
        raise ValueError("Data must contain 'tweet' and 'class' columns.")

    X_text = df["tweet"].astype(str)
    y = df["class"]

    logging.info("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_vectorized = vectorizer.fit_transform(X_text)

    logging.info(f"TF-IDF feature matrix shape: {X_vectorized.shape}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define paths
    features_path = os.path.join(output_dir, "X_features.npz")
    labels_path = os.path.join(output_dir, "y_labels.npy")
    vectorizer_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")

    # Save features and labels
    sparse.save_npz(features_path, X_vectorized)
    np.save(labels_path, y)
    joblib.dump(vectorizer, vectorizer_path)

    logging.info(f"Saved features to {features_path}")
    logging.info(f"Saved labels to {labels_path}")
    logging.info(f"Saved vectorizer to {vectorizer_path}")
    logging.info("Feature engineering complete.")


if __name__ == "__main__":
    input_file = os.path.abspath(
        os.path.join("..", "data", "processed", "train_combined.csv")
    )
    output_dir = os.path.abspath(os.path.join("..", "data", "features"))
    generate_features(input_file, output_dir)
