from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files["audio"]
    # Process the audio (speech-to-text, sentiment analysis, etc.)
    
    # Dummy response (replace with actual processing logic)
    response_data = {
        "transcription": "Hello, how can I help you?",
        "sentiment": "Positive",
        "response_time": 2.5,
        "accuracy": 92,
        "alerts": "None"
    }

    return jsonify(response_data)

if __name__ == "__main__":
    app.run(port=3000,debug=True)
