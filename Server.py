from flask import Flask, render_template, request, jsonify
from flask_mongoengine import MongoEngine
from models.user_model import User

app = Flask(__name__)

# Configure MongoDB
app.config["MONGODB_SETTINGS"] = {
    "db": "voiceAnalysisDB",
    "host": "localhost",
    "port": 27017
}

# Initialize Database
db = MongoEngine(app)

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

# Route to Add a New User
@app.route("/add_user", methods=["POST"])
def add_user():
    data = request.json
    if not data or "name" not in data or "phone" not in data:
        return jsonify({"error": "Missing name or phone number"}), 400

    if User.objects(phone=data["phone"]):
        return jsonify({"error": "Phone number already exists"}), 400

    new_user = User(name=data["name"], phone=data["phone"])
    new_user.save()

    return jsonify({"message": "User added successfully!", "user": {"name": new_user.name, "phone": new_user.phone}})

# Route to Get All Users
@app.route("/users", methods=["GET"])
def get_users():
    users = User.objects()
    users_list = [{"name": user.name, "phone": user.phone} for user in users]
    return jsonify({"users": users_list})

if __name__ == "__main__":
    app.run(port=3000, debug=True)