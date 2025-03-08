from flask import Flask, render_template, request, jsonify
# from mongoengine import connect
# from models.user_model import User

app = Flask(__name__)

# Connect MongoDB
# connect(db="voiceAnalysisDB", host="localhost", port=27017)

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/upload", methods=["POST"])
# def upload():
#     if "audio" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     audio_file = request.files["audio"]
#     # Process the audio (speech-to-text, sentiment analysis, etc.)
    
#     # Dummy response (replace with actual processing logic)
#     response_data = {
#         "transcription": "Hello, how can I help you?",
#         "sentiment": "Positive",
#         "response_time": 2.5,
#         "accuracy": 92,
#         "alerts": "None"
#     }

#     return jsonify(response_data)


# @app.route("/add_user", methods=["POST"])
# def add_user():
#     data = request.json
#     if not data or "phone" not in data:
#         return jsonify({"error": "Missing phone number"}), 400  # Name is no longer required

#     if User.objects(phone=data["phone"]):
#         return jsonify({"error": "Phone number already exists"}), 400

#     new_user = User(
#         name=data.get("name", ""),  # Default to empty string if name is missing
#         phone=data["phone"]
#     )
#     new_user.save()

#     return jsonify({"message": "User added successfully!", "user": {"name": new_user.name, "phone": new_user.phone}})

# Route to Get All Users
# @app.route("/users", methods=["GET"])
# def get_users():
#     users = User.objects()
#     users_list = [{"name": user.name, "phone": user.phone} for user in users]
#     return jsonify({"users": users_list})

@app.route('/')
def home():
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)

# if __name__ == "__main__":
#     app.run(port=3000, debug=True)
