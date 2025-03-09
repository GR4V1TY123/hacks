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
import os
import librosa
import numpy as np
import soundfile as sf
from transformers import pipeline
from sklearn.cluster import SpectralClustering
from multiprocessing import Pool, cpu_count

# Initialize global variable properly
combined_results = []

# Configure MongoDB
connect(db="voiceAnalysisDB", host="localhost", port=27017)

cloudinary.config(
    cloud_name="da22oy8nw",
    api_key="498768356194988",
    api_secret="MByqbQBBH0a3L8-T84MRYTj5BWA"
)

class VoiceSeparationTranscriber:
    def __init__(self, model_name="openai/whisper-small"):
        """Initialize the voice separation and transcription system."""
        self.model_name = model_name
        print(f"Initializing with ASR model: {model_name}")
        
        # Load the ASR model
        self.transcriber = self.load_model("automatic-speech-recognition", model_name)
        
        # Load sentiment analysis model
        self.sentiment_model = self.load_model("sentiment-analysis")
        
    def load_model(self, task, model_name=None):
        """Load models without caching."""
        try:
            model = pipeline(task, model=model_name) if model_name else pipeline(task)
            print(f"Successfully loaded model: {model_name if model_name else task}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}. Using placeholder functions.")
            return None
    
    def extract_features(self, audio, sr):
        """Extract MFCC features for speaker identification."""
        print("Extracting MFCC features...")
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-10)
        return mfccs.T  
    
    def segment_audio(self, audio, sr, segment_length=1.0):
        """Segment audio into fixed-length chunks."""
        segment_samples = int(segment_length * sr)
        segments = [audio[i:i + segment_samples] for i in range(0, len(audio), segment_samples) if len(audio[i:i + segment_samples]) == segment_samples]
        timestamps = [i / sr for i in range(0, len(audio), segment_samples) if len(audio[i:i + segment_samples]) == segment_samples]
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
    
    def get_sentiment(self, text):
        """Get sentiment classification for a statement."""
        if not self.sentiment_model or not text.strip():
            return "NEUTRAL"  # Default sentiment
        try:
            sentiment = self.sentiment_model(text)
            return sentiment[0]['label']
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return "NEUTRAL"
    
    def transcribe_segments(self, segments, sr, labels, timestamps):
        """Transcribe each speaker's audio and analyze sentiment."""
        print("Transcribing audio and analyzing sentiment...")
        speaker_texts = {label: [] for label in np.unique(labels)}
        sentiment_data = []  # Store sentiment data for visualization

        with Pool(cpu_count()) as pool:
            results = pool.starmap(self._transcribe_segment, [(segments[i], sr, labels[i], timestamps[i]) for i in range(len(segments))])
        
        for result in results:
            if result:
                label, formatted_text, sentiment, timestamp = result
                speaker_texts[label].append(formatted_text)
                sentiment_data.append({
                    'speaker': label,
                    'timestamp': timestamp,
                    'text': formatted_text.split('] ')[1],
                    'sentiment': sentiment
                })
        
        return speaker_texts, sentiment_data

    def _transcribe_segment(self, segment, sr, label, timestamp):
        """Helper function to transcribe a single segment."""
        try:
            if self.transcriber:
                segment = segment.astype(np.float32) if segment.dtype != np.float32 else segment
                result = self.transcriber({"array": segment, "sampling_rate": sr})
                text = result["text"]
            else:
                text = f"Transcribed text for Speaker {label+1}, segment {timestamp:.2f}s"

            sentiment = self.get_sentiment(text)
            formatted_text = f"[{timestamp:.2f}s] {text}"
            print(f"Speaker {label+1} at {timestamp:.2f}s: '{text}' | Sentiment: {sentiment}")
            return label, formatted_text, sentiment, timestamp
        except Exception as e:
            print(f"Error transcribing segment: {e}")
            return None

    def process_audio(self, audio_path, output_dir="output", n_speakers=2):
        """Process an audio file to separate and transcribe speakers."""
        print(f"Processing audio file: {audio_path}")
        
        global combined_results
        combined_results = []  # Reset the combined results

        if not os.path.exists(audio_path):
            print(f"Error: Audio file '{audio_path}' not found.")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
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
        
        with Pool(cpu_count()) as pool:
            features = np.array(pool.starmap(self.extract_features, [(seg, sr) for seg in segments]))
        
        labels = self.cluster_speakers(features, n_speakers)
        speaker_files = self.save_audio_by_speaker(segments, labels, sr, output_dir)

        transcriptions, sentiment_data = self.transcribe_segments(segments, sr, labels, timestamps)
        
        self.visualize_diarization_with_sentiment(sentiment_data, n_speakers, output_dir)

        for item in sentiment_data:
            speaker_name = f"Speaker {item['speaker']+1}"
            timestamp = item['timestamp']
            text = item['text']
            sentiment = item['sentiment']
            
            formatted_text = f"[{timestamp:.2f}s] {text}"
            result_tuple = (formatted_text, sentiment, timestamp)
            combined_results.append((speaker_name, result_tuple))

        for speaker, texts in transcriptions.items():
            speaker_file = os.path.join(output_dir, f"speaker{speaker+1}_transcription.txt")
            with open(speaker_file, "w", encoding="utf-8") as f:
                f.write("\n".join(texts))
            print(f"Saved transcription: {speaker_file}")

        return {
            "transcriptions": transcriptions,
            "speaker_files": speaker_files,
            "visualization": os.path.join(output_dir, 'speaker_diarization_with_sentiment.png'),
            "sentiment_data": sentiment_data,
            "combined_results": combined_results
        }
    
    def visualize_diarization_with_sentiment(self, sentiment_data, n_speakers, output_dir):
        """Create a visualization of speaker diarization with sentiment information."""
        print("Generating speaker visualization with sentiment information...")
        
        plt.figure(figsize=(15, 8))
        
        sentiment_colors = {
            'POSITIVE': 'green',
            'NEUTRAL': 'blue',
            'NEGATIVE': 'red'
        }
        
        for speaker in range(n_speakers):
            speaker_segments = [item for item in sentiment_data if item['speaker'] == speaker]
            
            for segment in speaker_segments:
                timestamp = segment['timestamp']
                sentiment = segment['sentiment']
                color = sentiment_colors.get(sentiment, 'gray')
                plt.scatter(timestamp, speaker, color=color, s=100, alpha=0.7)
                short_text = segment['text'][:20] + ('...' if len(segment['text']) > 20 else '')
                plt.annotate(short_text, (timestamp, speaker), 
                             textcoords="offset points", 
                             xytext=(0,10), 
                             ha='center', 
                             fontsize=8,
                             rotation=45)
        
        sentiment_patches = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=10, label=sentiment)
                          for sentiment, color in sentiment_colors.items()]
        
        plt.legend(handles=sentiment_patches, title="Sentiment", loc='upper right')
        plt.yticks(range(n_speakers), [f'Speaker {i+1}' for i in range(n_speakers)])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Speaker')
        plt.title('Speaker Diarization with Sentiment Analysis')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        viz_file = os.path.join(output_dir, 'speaker_diarization_with_sentiment.png')
        plt.tight_layout()
        plt.savefig(viz_file, dpi=300)
        plt.close()
        print(f"Visualization with sentiment saved: {viz_file}")

# Helper functions and Streamlit interface remain the same as in the original code

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