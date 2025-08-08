import gradio as gr
import joblib
import os

# Load the saved model and vectorizer
model_path =os.path.join(os.path.dirname(__file__),"..","models","model.pkl")
# model_path =os.path.join(os.path.dirname(__file__),"..","models","vectorizer.pkl")


vectorizer, model  = joblib.load(model_path)
# vectorizer = joblib.load(vectorizer_path)

def predict(tweet_text):
    # Vectorize the raw tweet text
    X_vec = vectorizer.transform([tweet_text])
    
    # Predict the class
    prediction = model.predict(X_vec)[0]

    # Map numeric class to human-readable label if needed
    label_map = {0: "Not Abusive", 1: "Abusive"}
    return label_map.get(prediction, str(prediction))

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Enter tweet text here..."),
    outputs="text",
    title="Tweet Abuse Detection",
    description="Enter a tweet and the model predicts if it is abusive or not."
)

if __name__ == "__main__":
    iface.launch()
