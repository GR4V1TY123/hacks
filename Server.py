from flask import Flask, render_template, request, jsonify
from mongoengine import connect
import cloudinary
import cloudinary.uploader
import matplotlib
matplotlib.use("Agg")  # Fix Matplotlib GUI warning
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import io
import base64
from models.user_model import User
import random
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from flask import send_file
from reportlab.lib.utils import ImageReader

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



def generate_pie_chart(sentiment):
    """Generates a pie chart for sentiment analysis and saves it."""
    labels = sentiment.keys()
    sizes = sentiment.values()
    colors = ["green", "red", "gray"]

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
    plt.title("Sentiment Analysis")

    chart_path = "static/sentiment_chart.png"
    plt.savefig(chart_path)  # Save as image
    plt.close()
    return chart_path

@app.route("/upload", methods=["POST"])
def upload():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    # Simulated Analysis Data (Replace with real ML analysis)
    transcription = "Hello, this is a test transcription. It was previously believed that mangoes originated from a single domestication event in South Asia before being spread to Southeast Asia, but a 2019 study found no evidence of a center of diversity in India. Instead, it identified a higher unique genetic diversity in Southeast Asian cultivars than in Indian cultivars, indicating that mangoes may have originally been domesticated first in Southeast Asia before being introduced to South Asia. However, the authors also cautioned that the diversity in Southeast Asian mangoes might be the result of other reasons (like interspecific hybridization with other Mangifera species native to the Malesian ecoregion). Nevertheless, the existence of two distinct genetic populations also identified by the study indicates that the domestication of the mango is more complex than previously assumed and would at least indicate multiple domestication events in Southeast Asia and South Asia.[1][2]"
    response_time = 20
    accuracy = round(random.uniform(85, 99), 2)
    alerts = "None"
    audio_file = request.files["audio"]
    sentiment = {"Positive": 50, "Negative": 30, "Neutral": 20}
    chart_path = generate_pie_chart(sentiment)
    upload_result = cloudinary.uploader.upload(audio_file, resource_type="auto")
    audio_url = upload_result["secure_url"]

    return jsonify({
    "transcription": transcription,
    "sentiment": sentiment,
    "response_time": response_time,
    "accuracy": accuracy,
    "alerts": alerts,
    "audio_url": audio_url,  # Include uploaded audio URL
    "chart_path": chart_path
})
    



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

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("Audio Analysis Report")

    y_position = 750  # Start from the top

    # Title
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(100, y_position, "Audio Analysis Report")
    y_position -= 30

    # Wrap text function
    def draw_wrapped_text(pdf, text, x, y, max_width=500, line_height=15):
        """Splits long text into lines that fit within the max width."""
        from textwrap import wrap
        lines = wrap(text, width=60)  # Adjust width as needed
        for line in lines:
            pdf.drawString(x, y, line)
            y -= line_height
        return y

    # Content
    pdf.setFont("Helvetica", 12)
    text_entries = {
        "Transcription": data["transcription"],
        "Sentiment": data["sentiment"],
        "Response Time": f"{data['response_time']} sec",
        "Accuracy": f"{data['accuracy']}%",
        "Critical Alerts": data["alerts"]
    }

    for label, value in text_entries.items():
        y_position = draw_wrapped_text(pdf, f"{label}: {value}", 100, y_position, max_width=400)
        y_position -= 10  # Extra spacing between items

    # 🎵 Add clickable audio link
    pdf.setFillColorRGB(0, 0, 1)  # Blue color for link
    pdf.drawString(100, y_position, "🔗 Listen to Audio Report")
    pdf.linkURL(data["audio_url"], (100, y_position - 10, 350, y_position + 10))
    y_position -= 40  # Extra space before chart

    # 📊 Add sentiment analysis chart dynamically below text
    sentiment_data = {"Positive": 50, "Negative": 30, "Neutral": 20}  # Example data
    chart_buffer = generate_pie_chart(sentiment_data)
    chart_reader = ImageReader(chart_buffer)

    chart_y_position = max(y_position - 300, 50)  # Ensure it stays on the page
    pdf.drawImage(chart_reader, 100, chart_y_position, width=300, height=300)

    # Save the PDF
    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="audio_report.pdf", mimetype="application/pdf")



@app.route('/')
def home():
    return render_template('index.html')

if __name__=='__main__':
    app.run(port=3000,debug=True)

# if __name__ == "__main__":
#     app.run(port=3000, debug=True)
