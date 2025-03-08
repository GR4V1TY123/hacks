import whisper
import pyaudio
import numpy as np
import os
import wave

def transcribe_chunk(model, chunk_file):
    result = model.transcribe(chunk_file)
    return result['text']
# Load the Whisper model
def record_chunk(p,stream,chunk_file,chunk_length=1):
    frames=[]
    for _ in range(0,int(16000/1024*chunk_length)):
        data=stream.read(1024)
        frames.append(data)

    with wave.open(chunk_file, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))  # Sample width
        wf.setframerate(16000)  # 16 kHz sample rate
        wf.writeframes(b''.join(frames))

def main2():
    model = whisper.load_model("base")
    p=pyaudio.PyAudio()
    stream=p.open(format=pyaudio.paInt16,channels=1,rate=16000,input=True,frames_per_buffer=1024)
    accumulated_transcription=""

    try:
        while True:
            chunk_file="temp_chunk.wav"
            record_chunk(p,stream,chunk_file)
            transcription=transcribe_chunk(model,chunk_file)
            print(transcription)
            os.remove(chunk_file)

            accumulated_transcription+=transcription + " "
    except KeyboardInterrupt:
        print("Stopping....")
        with open("log.txt","w") as log_file:
            log_file.write(accumulated_transcription)
    finally:
        print("LOG:"+accumulated_transcription)
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__=="__main__":
    main2()