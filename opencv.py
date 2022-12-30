from flask import Flask, request
import tensorflow as tf

# Load the TensorFlow Hub model for sentiment analysis
model = tf.hub.load("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1")

app = Flask(__name__)

@app.route("/classify_sentiment", methods=["POST"])
def classify_sentiment():
    # Get the text to be classified from the request
    text = request.form["text"]

    # Encode the text as a numerical representation
    encoded_text = model(text)

    # Classify the sentiment of the text
    prediction = tf.argmax(encoded_text, axis=-1)

    # Return the predicted sentiment as a response
    return str(prediction)

if __name__ == "__main__":
    app.run()
