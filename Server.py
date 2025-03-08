from flask import Flask, render_template, request, jsonify, send_file
from mongoengine import connect
import cloudinary
import cloudinary.uploader
from models.user_model import User
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

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

    # Dummy response (replace with actual processing logic)/
    response_data = {
        "transcription": "Hello, how can I help you?",
        "sentiment": "Positive",
        "response_time": 2.5,
        "accuracy": 92,
        "alerts": "None",
        "audio_url": audio_url  # Returning the uploaded audio file URL
    }

    return jsonify(response_data)

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

    return jsonify({"message": "User added successfully!", "user": {"name": new_user.name, "phone": new_user.phone}})

# Route to Get All Users
@app.route("/users", methods=["GET"])
def get_users():
    users = User.objects()
    users_list = [{"name": user.name, "phone": user.phone, "audio_url": user.audio_url} for user in users]
    return jsonify({"users": users_list})

# 📝 Route to generate a PDF report
@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    data = request.json

    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Create a PDF in memory
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("Audio Analysis Report")

    # Title
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(100, 750, "Audio Analysis Report")

    # Content
    pdf.setFont("Helvetica", 12)
    pdf.drawString(100, 720, f"Transcription: {data['transcription']}")
    pdf.drawString(100, 700, f"Sentiment: {data['sentiment']}")
    pdf.drawString(100, 680, f"Response Time: {data['response_time']} sec")
    pdf.drawString(100, 660, f"Accuracy: {data['accuracy']}%")
    pdf.drawString(100, 640, f"Critical Alerts: {data['alerts']}")

    # 🎵 Add a clickable link to the Cloudinary audio file
    pdf.setFillColorRGB(0, 0, 1)  # Blue color for link
    pdf.drawString(100, 620, "🔗 Listen to Audio Report")
    pdf.linkURL(data["audio_url"], (100, 610, 350, 630))  # Link position

    # Save the PDF
    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="audio_report.pdf", mimetype="application/pdf")


@app.route('/')
def home():
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)

# if __name__ == "__main__":
#     app.run(port=3000, debug=True)
