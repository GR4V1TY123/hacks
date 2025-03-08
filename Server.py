from flask import Flask, render_template, request, jsonify
from mongoengine import connect
import cloudinary
import cloudinary.uploader
from models.user_model import User

app = Flask(__name__)

# Connect MongoDB
connect(db="voiceAnalysisDB", host="localhost", port=27017)

# Configure Cloudinary
cloudinary.config(
    cloud_name="da22oy8nw",
    api_key="498768356194988",
    api_secret="MByqbQBBH0a3L8-T84MRYTj5BWA"
)

# @app.route("/")
# def index():
#     return render_template("index.html")

# Route to Upload and Process Audio
@app.route("/upload", methods=["POST"])
def upload():
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files["audio"]

    # Upload to Cloudinary
    cloudinary_response = cloudinary.uploader.upload(audio_file, resource_type="video")  # Cloudinary treats audio as "video"

    audio_url = cloudinary_response["secure_url"]

    # Dummy response (replace with actual processing logic)
    response_data = {
        "transcription": "Hello, how can I help you?",
        "sentiment": "Positive",
        "response_time": 2.5,
        "accuracy": 92,
        "alerts": "None",
        "audio_url": audio_url  # Returning the uploaded audio file URL
    }

#     return jsonify(response_data)

# Route to Add a New User (Now phone is also optional)
@app.route("/add_user", methods=["POST"])
def add_user():
    data = request.json

    if not data:
        return jsonify({"error": "No data provided"}), 400

    new_user = User(
        name=data.get("name", ""),  # Default to empty string if name is missing
        phone=data.get("phone", ""),  # Default to empty string if phone is missing
    )
    new_user.save()

#     return jsonify({"message": "User added successfully!", "user": {"name": new_user.name, "phone": new_user.phone}})

# Route to Get All Users
@app.route("/users", methods=["GET"])
def get_users():
    users = User.objects()
    users_list = [{"name": user.name, "phone": user.phone, "audio_url": user.audio_url} for user in users]
    return jsonify({"users": users_list})

@app.route('/')
def home():
    return render_template('index.html')

if __name__=='__main__':
    app.run(port=3000,debug=True)

# if __name__ == "__main__":
#     app.run(port=3000, debug=True)
