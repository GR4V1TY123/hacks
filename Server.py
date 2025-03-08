from flask import Flask, render_template, request, jsonify
from mongoengine import connect
import cloudinary
import cloudinary.uploader
import matplotlib
matplotlib.use("Agg")  # Fix Matplotlib GUI warning
import matplotlib.pyplot as plt
import io
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

sentiment = {}

def generate_data():
    generated_data = []
    sentiment_data = [
    {"sentence": "One", "sentiment": "Neutral"},
    {"sentence": "Can we go for a swim in the sea? Two", "sentiment": "Neutral"},
    {"sentence": "It's a beautiful day in the south of England", "sentiment": "Positive"},
    {"sentence": "Three", "sentiment": "Neutral"}
]
    for entry in sentiment_data:
        alert = "None"  # Default value
        if entry["sentiment"] not in ["Positive", "Neutral"]:
            alert = "Set (Red Flag)"  # If sentiment is not Positive or Neutral, set alert to "Red Flag"
        
        generated_data.append({
            "transcription": entry["sentence"],  # Sentence as transcription
            "sentiment": entry["sentiment"],      # Sentiment
            "response_time": entry["resp_time"],  # Response time
            "alerts": alert                 # Alert based on sentiment
        })
    
    # Return the generated data as a JSON response
    return (generated_data)

def calculate_sentiment_percentage(sentiment_data):
    # Initialize counters for each sentiment
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    
    # Count the occurrences of each sentiment
    total_sentences = len(sentiment_data)
    for entry in sentiment_data:
        sentiment = entry['sentiment']
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
    
    # Calculate percentages
    sentiment_percentages = {sentiment: (count / total_sentences) * 100 for sentiment, count in sentiment_counts.items()}
    
    return sentiment_percentages


# Example input
sentiment_data = [
    {"sentence": "One", "sentiment": "Neutral"},
    {"sentence": "Can we go for a swim in the sea? Two", "sentiment": "Neutral"},
    {"sentence": "It's a beautiful day in the south of England", "sentiment": "Positive"},
    {"sentence": "Three", "sentiment": "Neutral"}
]




# Calculate the sentiment percentages
percentages = calculate_sentiment_percentage(sentiment_data)

def generate_pie_chart(sentiment_dict):
    """Generates a pie chart for sentiment analysis and saves it."""
    labels = sentiment_dict.keys()
    sizes = sentiment_dict.values()
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

    # transcription = """From tropical Asia, mangoes were introduced to East Africa by Arab and Persian traders in the ninth to tenth centuries.[20] The 14th-century Moroccan traveler Ibn Battuta reported it at Mogadishu.[21] It was spread further into other areas around the world during the Colonial Era. The Portuguese Empire spread the mango from their colony in Goa to East and West Africa. From West Africa, they introduced it to Brazil from the 16th to the 17th centuries. From Brazil, it spread northwards to the Caribbean and eastern Mexico by the mid to late 18th century. The Spanish Empire also introduced mangoes directly from the Philippines to western Mexico via the Manila galleons from at least the 16th century. Mangoes were only introduced to Florida by 1833"""
    # response_time = 20
    # accuracy = round(random.uniform(85, 99), 2)
    # alerts = "None"
    audio_file = request.files["audio"] 

    # Sentiment data as a dictionary
    # sentiment = {
    #     "Positive": 25,
    #     "Negative": 25,
    #     "Neutral": 50
    # }
    
    sentiment=calculate_sentiment_percentage(sentiment_data)

    # Generate pie chart
    chart_path = generate_pie_chart(sentiment)

    # Upload audio file to Cloudinary
    upload_result = cloudinary.uploader.upload(audio_file, resource_type="auto")
    audio_url = upload_result["secure_url"]
    
    data=generate_data();

    
    return jsonify(data)
    



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
    users_list = [{"name": user.name, "phone": user.phone} for user in users]
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
    sentiment_data = data["sentiment"]  # Example data
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
