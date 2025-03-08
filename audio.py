import whisper

# Load the model (smallest one for speed)
model = whisper.load_model("tiny")

# Transcribe the audio file
result = model.transcribe("test1.mp3")  # Replace with your file

# Print the transcribed text    
print(result["text"])