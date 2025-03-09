import whisper
import os
from pydub import AudioSegment
import time
import threading as th
import json
import warnings
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Configure Gemini API
genai.configure(api_key='AIzaSyC3vNkSnEJl-eFloSm9M4Bw0F_cJv2vusY')
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Load Whisper model
whisper_model = whisper.load_model("base")

# Global variables
results = []
processing_done = False

# Streamlit UI
st.title("🎙️ Real-Time Audio Sentiment Analysis")
uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])

def printer():
    """Continuously updates the results in Streamlit."""
    index = 0
    result_placeholder = st.empty()  # Placeholder for dynamic updates

    while not processing_done or index < len(results):
        if index < len(results):
            chunk_text = f"**Chunk {index+1}:** [{results[index][1]}] {results[index][0]} (⏱ {results[index][2]}s)"
            result_placeholder.write(chunk_text)
            index += 1
        time.sleep(5)  # Check every 5 seconds

def split_audio(file_path, chunk_length=5000):
    """Splits audio into 5-second chunks."""
    audio = AudioSegment.from_file(file_path)
    chunk_files = []
    
    for i, start_time in enumerate(range(0, len(audio), chunk_length)):
        chunk = audio[start_time:start_time + chunk_length]
        chunk_filename = f"chunk_{i}.wav"
        chunk.export(chunk_filename, format="wav")
        chunk_files.append(chunk_filename)
    
    return chunk_files

def transcribe_chunk(chunk_file):
    """Transcribes a given audio chunk using Whisper."""
    result = whisper_model.transcribe(chunk_file)
    return result['text'].strip()

def get_sentiment(statement):
    """Gets sentiment classification from Gemini API."""
    if not statement.strip():
        return "Neutral"
    
    question = f"Classify the sentiment of this statement as Positive, Negative, or Neutral. Respond with only one word: '{statement}'"
    response = gemini_model.generate_content(question)
    return response.text.strip() if response else "Unknown"

def main():
    global processing_done

    if uploaded_file is not None:
        # Save uploaded file
        input_audio = "uploaded_audio.mp3"
        with open(input_audio, "wb") as f:
            f.write(uploaded_file.read())

        st.write("🔹 Splitting audio into chunks...")
        chunk_files = split_audio(input_audio)

        st.write("\n🔹 Transcribing and analyzing sentiment...")
        start_time = time.perf_counter()

        # Start real-time result display
        t1 = th.Thread(target=printer, daemon=True)
        t1.start()

        # Process all chunks and store results
        for chunk_file in chunk_files:
            transcription = transcribe_chunk(chunk_file)
            sentiment = get_sentiment(transcription)
            elapsed_time = round(time.perf_counter() - start_time, 2)
            results.append((transcription, sentiment, elapsed_time))
            os.remove(chunk_file)  # Cleanup chunk file

        processing_done = True  # Mark processing as done

        # Save results to JSON file
        with open("output.json", "w") as json_file:
            json.dump(results, json_file, indent=4)

        st.success("✅ Analysis completed! Results saved to `output.json`.")
        st.json(results)  # Display final JSON output

if __name__ == "__main__":
    main()
