import numpy as np
import librosa
import os
import torch
from sklearn.cluster import SpectralClustering
from transformers import pipeline
import matplotlib.pyplot as plt
import soundfile as sf

class VoiceSeparationTranscriber:
    def __init__(self, model_name="openai/whisper-small"):
        """Initialize the voice separation and transcription system."""
        self.model_name = model_name
        print(f"Initializing with ASR model: {model_name}")
        
        # Load the ASR model
        try:
            self.transcriber = pipeline("automatic-speech-recognition", model=model_name)
            print(f"Successfully loaded ASR model: {model_name}")
        except Exception as e:
            print(f"Error loading ASR model: {e}. Using placeholder transcriptions.")
            self.transcriber = None
        
    def extract_features(self, audio, sr):
        """Extract MFCC features for speaker identification."""
        print("Extracting MFCC features...")
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-10)
        return mfccs.T  
    
    def segment_audio(self, audio, sr, segment_length=1.0):
        """Segment audio into fixed-length chunks."""
        segment_samples = int(segment_length * sr)
        segments, timestamps = [], []
        
        for i in range(0, len(audio), segment_samples):
            segment = audio[i:i + segment_samples]
            if len(segment) == segment_samples:
                segments.append(segment)
                timestamps.append(i / sr)
        
        return segments, timestamps, segment_length
    
    def cluster_speakers(self, features, n_speakers=2):
        """Cluster audio segments into speaker groups."""
        print(f"Clustering into {n_speakers} speaker groups...")
        clustering = SpectralClustering(n_clusters=n_speakers, assign_labels="discretize",
                                        random_state=42, affinity='nearest_neighbors')
        
        try:
            labels = clustering.fit_predict(features)
            return labels
        except Exception as e:
            print(f"Clustering error: {e}. Using random labels as fallback.")
            return np.random.randint(0, n_speakers, size=len(features))
    
    def save_audio_by_speaker(self, segments, labels, sr, output_dir):
        """Save separated audio files for each speaker."""
        print("Saving audio by speaker...")
        speaker_files = {}
        for i, label in enumerate(np.unique(labels)):
            speaker_audio = np.concatenate([segments[j] for j in range(len(labels)) if labels[j] == label])
            speaker_file = os.path.join(output_dir, f"speaker{label+1}_audio.wav")
            sf.write(speaker_file, speaker_audio, sr)
            speaker_files[label] = speaker_file
            print(f"Saved Speaker {label+1} audio to {speaker_file}")
        
        return speaker_files
    
    def transcribe_segments(self, segments, sr, labels, timestamps):
        """Transcribe each speaker's audio."""
        print("Transcribing audio...")
        speaker_texts = {}

        for i, label in enumerate(labels):
            if label not in speaker_texts:
                speaker_texts[label] = []

            try:
                if self.transcriber:
                    segment = segments[i].astype(np.float32) if segments[i].dtype != np.float32 else segments[i]
                    result = self.transcriber({"array": segment, "sampling_rate": sr})
                    text = result["text"]
                else:
                    text = f"Transcribed text for Speaker {label+1}, segment {i+1}"

                timestamp = f"[{timestamps[i]:.2f}s]"
                formatted_text = f"{timestamp} {text}"
                speaker_texts[label].append(formatted_text)
                
                # Print to console
                print(f"Speaker {label+1}: {formatted_text}")

            except Exception as e:
                print(f"Error transcribing segment {i} for speaker {label}: {e}")
                speaker_texts[label].append(f"[Error in segment {i}]")
        
        return speaker_texts
    
    def process_audio(self, audio_path, output_dir="output", n_speakers=2):
        """Process an audio file to separate and transcribe speakers."""
        print(f"Processing audio file: {audio_path}")

        if not os.path.exists(audio_path):
            print(f"Error: Audio file '{audio_path}' not found.")
            return None
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("Loading audio file...")
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            print(f"Audio loaded: {len(audio)/sr:.2f} seconds at {sr}Hz")
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None
        
        segments, timestamps, _ = self.segment_audio(audio, sr)
        print(f"Audio segmented into {len(segments)} segments")

        if not segments:
            print("No complete segments found.")
            return None
        
        features = np.array([np.mean(self.extract_features(seg, sr), axis=0) for seg in segments])
        labels = self.cluster_speakers(features, n_speakers)
        speaker_files = self.save_audio_by_speaker(segments, labels, sr, output_dir)

        print("Generating speaker visualization...")
        plt.figure(figsize=(10, 4))
        for i in range(n_speakers):
            speaker_times = [timestamps[j] for j in range(len(labels)) if labels[j] == i]
            plt.plot(speaker_times, [i] * len(speaker_times), 'o', label=f'Speaker {i+1}')
        plt.yticks(range(n_speakers))
        plt.xlabel('Time (seconds)')
        plt.ylabel('Speaker')
        plt.title('Speaker Diarization')
        plt.legend()
        viz_file = os.path.join(output_dir, 'speaker_diarization.png')
        plt.savefig(viz_file)
        plt.close()
        print(f"Visualization saved: {viz_file}")

        transcriptions = self.transcribe_segments(segments, sr, labels, timestamps)

        for speaker, texts in transcriptions.items():
            speaker_file = os.path.join(output_dir, f"speaker{speaker+1}_transcription.txt")
            with open(speaker_file, "w", encoding="utf-8") as f:
                f.write("\n".join(texts))
            print(f"Saved transcription: {speaker_file}")

        return {"transcriptions": transcriptions, "speaker_files": speaker_files, "visualization": viz_file}

if __name__ == "__main__":
    audio_file_path = "twowaychat.mp3"  # Update with your actual file
    output_directory = "output"
    num_speakers = 2
    model_name = "openai/whisper-small"

    transcriber = VoiceSeparationTranscriber(model_name=model_name)
    result = transcriber.process_audio(audio_file_path, output_directory, num_speakers)

    if result:
        print("\n✅ Processing complete! Results saved in 'output/'")
    else:
        print("\n❌ Error processing the audio file.")
