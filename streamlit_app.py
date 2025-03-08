import streamlit as st
import random
import matplotlib
import cloudinary
import cloudinary.uploader
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import io
from mongoengine import connect

# Configure MongoDB
connect(db="voiceAnalysisDB", host="localhost", port=27017)

# Configure Cloudinary
cloudinary.config(
    cloud_name="da22oy8nw",
    api_key="498768356194988",
    api_secret="MByqbQBBH0a3L8-T84MRYTj5BWA"
)

# Helper functions
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
            "response_time": random.randint(10, 30),  # Random response time for now
            "alerts": alert                      # Alert based on sentiment
        })
    return generated_data

def calculate_sentiment_percentage(sentiment_data):
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    total_sentences = len(sentiment_data)
    for entry in sentiment_data:
        sentiment = entry['sentiment']
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
    sentiment_percentages = {sentiment: (count / total_sentences) * 100 for sentiment, count in sentiment_counts.items()}
    return sentiment_percentages

def generate_pie_chart(sentiment_dict):
    labels = sentiment_dict.keys()
    sizes = sentiment_dict.values()
    colors = ["green", "red", "gray"]

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
    plt.title("Sentiment Analysis")

    chart_path = "sentiment_chart.png"
    plt.savefig(chart_path)  # Save as image
    plt.close()
    return chart_path

# Streamlit Interface
st.title("Audio Sentiment Analysis")

# File Upload
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])

if audio_file:
    st.audio(audio_file)

    # Sentiment Data and Calculations
    sentiment_data = generate_data()
    sentiment_percentages = calculate_sentiment_percentage(sentiment_data)

    # Display Sentiment Percentages
    st.write("Sentiment Analysis Percentages:")
    st.write(sentiment_percentages)

    # Generate and Display Pie Chart
    chart_path = generate_pie_chart(sentiment_percentages)
    st.image(chart_path, caption="Sentiment Analysis Chart")

    # Generate PDF
    if st.button("Generate PDF Report"):
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        pdf.setTitle("Audio Analysis Report")

        y_position = 750

        # Title
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(100, y_position, "Audio Analysis Report")
        y_position -= 30

        # Content
        pdf.setFont("Helvetica", 12)
        text_entries = {
            "Transcription": "This is a sample transcription.",
            "Sentiment": str(sentiment_percentages),
            "Response Time": f"20 sec",  # Placeholder
            "Accuracy": f"{random.randint(85, 99)}%",
            "Critical Alerts": "None"
        }

        for label, value in text_entries.items():
            pdf.drawString(100, y_position, f"{label}: {value}")
            y_position -= 20

        # Add Sentiment Chart to PDF
        chart_reader = ImageReader(chart_path)
        pdf.drawImage(chart_reader, 100, y_position - 200, width=300, height=300)

        # Save PDF to buffer
        pdf.showPage()
        pdf.save()
        buffer.seek(0)

        # Send the PDF to user
        st.download_button(
            label="Download PDF Report",
            data=buffer,
            file_name="audio_report.pdf",
            mime="application/pdf"
        )
else:
    st.write("Please upload an audio file to proceed.")