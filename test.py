import whisper
import os
from pydub import AudioSegment
import google.generativeai as genai
import time
import threading as th
import json
import warnings
import time 
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


results = []
processing_done = False  # Flag to indicate when processing is done

def printer():
    """ Continuously prints results every 5 seconds """
    index = 0
    while not processing_done or index < len(results):
        if index < len(results):  # Print only if a new result is available
            print(f"Chunk {index+1}: [{results[index][1]}] {results[index][0]}")
            index += 1
        time.sleep(5)  # Wait before checking again

# Configure Gemini API
genai.configure(api_key="AIzaSyC3vNkSnEJl-eFloSm9M4Bw0F_cJv2vusY")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Load Whisper model
whisper_model = whisper.load_model("medium")

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
    current_time=time.time()
    input_audio = "harvard.mp3"  # Change this to your file
    print("🔹 Splitting audio into chunks...")
    chunk_files = split_audio(input_audio)

    print("\n🔹 Transcribing and analyzing sentiment...")
    t1 = th.Thread(target=printer, daemon=True)
    t1.start()

    # Process all chunks and store results
    for chunk_file in chunk_files:
        transcription = transcribe_chunk(chunk_file)
        sentiment = get_sentiment(transcription)
        results.append((transcription, sentiment, round(time.time()-current_time,2)))
        os.remove(chunk_file)  # Cleanup chunk file

    processing_done = True  # Mark processing as done

    # Wait a bit to ensure all results are printed before exiting
    time.sleep(5)
    t1.join()
    json_data=json.dumps(results)
    print(json_data)
if __name__ == "__main__":
    main()
