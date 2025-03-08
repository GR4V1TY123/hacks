import streamlit as st
import random
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import io
from mongoengine import connect
import cloudinary
import cloudinary.uploader

# Configure MongoDB
connect(db="voiceAnalysisDB", host="localhost", port=27017)

cloudinary.config(
    cloud_name="da22oy8nw",
    api_key="498768356194988",
    api_secret="MByqbQBBH0a3L8-T84MRYTj5BWA"
)

# Helper functions
def generate_data():
    generated_data = []
    sentiment_data = [
        {"sentence": "One", "sentiment": "Negative"},
        {"sentence": "Can we go for a swim in the sea? Two", "sentiment": "Negative"},
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
    # Upload the audio file to Cloudinary
    audio_file_path = audio_file.name
    upload_result = cloudinary.uploader.upload(audio_file, resource_type="video", folder="audios")  # Using resource_type="video" for audio files

    # Get the URL of the uploaded audio file
    audio_url = upload_result['secure_url']
    st.write(f"Audio uploaded successfully! You can access the file at: [Click Here]({audio_url})")

    # Sentiment Data and Calculations
    sentiment_data = generate_data()
    sentiment_percentages = calculate_sentiment_percentage(sentiment_data)

    # Display Sentiment Percentages
    st.write("Sentiment Analysis Percentages:")
    st.write(sentiment_percentages)
    
    # Display Transcriptions
    st.write("Transcript Analysis:")
    
    # Initialize the transcription HTML
    transcription_html = ""

    # Loop through sentiment data and generate transcription blocks
    for entry in sentiment_data:
        # Determine the border color based on sentiment
        if entry['sentiment'] == "Positive":
            border_color = "#28a745"  # Green for Positive
        elif entry['sentiment'] == "Negative":
            border_color = "#dc3545"  # Red for Negative
        else:
            border_color = "#6c757d"  # Gray for Neutral

        transcription_html += f"""
        <div class="transcription" style="border-color: {border_color};">
            <strong>Transcription:</strong> {entry['transcription']}
            <br><strong>Sentiment:</strong> {entry['sentiment']}
            <br><strong>Alerts:</strong> {entry['alerts']}
            <br><strong>Response Time:</strong> {entry['response_time']} seconds
        </div>
        """

    # Add the CSS for transcription styling
    st.markdown(
        f"""
        <style>
            .transcription {{
                font-family: 'Arial', sans-serif;
                font-size: 18px;
                color: #4A90E2;
                padding: 10px;
                border-radius: 8px;
                border: 2px solid;
                box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.1);
                margin-top: 10px;
                margin-bottom: 10px;
            }}
        </style>
        {transcription_html}
        """,
        unsafe_allow_html=True
    )

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

        # Iterate through sentiment data and add to PDF
        for entry in sentiment_data:
            # Check if we need to create a new page
            if y_position < 100:
                pdf.showPage()  # Start a new page
                y_position = 750  # Reset y_position to the top of the new page

            # Make key bold and add content
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(100, y_position, "Transcription:")
            y_position -= 15
            pdf.setFont("Helvetica", 12)  # Switch back to regular font
            pdf.drawString(100, y_position, entry['transcription'])
            y_position -= 15

            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(100, y_position, "Sentiment:")
            y_position -= 15
            pdf.setFont("Helvetica", 12)
            pdf.drawString(100, y_position, entry['sentiment'])
            y_position -= 15

            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(100, y_position, "Response Time:")
            y_position -= 15
            pdf.setFont("Helvetica", 12)
            pdf.drawString(100, y_position, f"{entry['response_time']} seconds")
            y_position -= 15

            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(100, y_position, "Alerts:")
            y_position -= 15
            pdf.setFont("Helvetica", 12)
            pdf.drawString(100, y_position, entry['alerts'])
            y_position -= 30

            # Ensure content fits within the page height
            if y_position < 100:
                pdf.showPage()
                y_position = 750

        # Add Sentiment Chart to PDF
        chart_reader = ImageReader(chart_path)
        pdf.drawImage(chart_reader, 100, y_position - 200, width=300, height=200)

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
