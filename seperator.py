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
        """
        Initialize the voice separation and transcription system.
        
        Args:
            model_name: The name of the ASR model to use
        """
        self.model_name = model_name
        print(f"Initializing with ASR model: {model_name}")
        # Initialize the ASR model
        try:
            self.transcriber = pipeline("automatic-speech-recognition", model=model_name)
            print(f"Successfully loaded ASR model: {model_name}")
        except Exception as e:
            print(f"Error loading ASR model: {e}")
            print("Using placeholder transcriptions instead.")
            self.transcriber = None
        
    def extract_features(self, audio, sr):
        """Extract MFCC features from audio for speaker identification."""
        print("Extracting MFCC features...")
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        
        # Normalize MFCCs
        mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-10)
        
        return mfccs.T  # Transpose to get time frames as rows
    
    def segment_audio(self, audio, sr, segment_length=1.0):
        """Segment audio into fixed-length chunks."""
        segment_samples = int(segment_length * sr)
        segments = []
        timestamps = []
        
        for i in range(0, len(audio), segment_samples):
            segment = audio[i:i + segment_samples]
            if len(segment) == segment_samples:  # Only use complete segments
                segments.append(segment)
                timestamps.append(i / sr)  # Start time in seconds
        
        return segments, timestamps, segment_length
    
    def cluster_speakers(self, features, n_speakers=2):
        """Cluster audio segments into speaker groups."""
        print(f"Clustering audio into {n_speakers} speaker groups...")
        
        # Use spectral clustering to separate speakers
        clustering = SpectralClustering(n_clusters=n_speakers, 
                                       assign_labels="discretize",
                                       random_state=42,
                                       affinity='nearest_neighbors')
        
        # Fit the clustering model
        try:
            labels = clustering.fit_predict(features)
            return labels
        except Exception as e:
            print(f"Clustering error: {e}. Using random assignments as fallback.")
            return np.random.randint(0, n_speakers, size=len(features))
    
    def save_audio_by_speaker(self, segments, labels, sr, output_dir):
        """Save audio segments by speaker."""
        print("Saving audio by speaker...")
        
        # Group segments by speaker
        speaker_segments = {}
        for i, label in enumerate(labels):
            if label not in speaker_segments:
                speaker_segments[label] = []
            speaker_segments[label].append(segments[i])
        
        # Save each speaker's audio
        speaker_files = {}
        for speaker, segs in speaker_segments.items():
            # Concatenate all segments for this speaker
            speaker_audio = np.concatenate(segs)
            
            # Save to file
            speaker_file = os.path.join(output_dir, f"speaker{speaker+1}_audio.wav")
            sf.write(speaker_file, speaker_audio, sr)
            
            speaker_files[speaker] = speaker_file
            print(f"Saved Speaker {speaker+1} audio to {speaker_file}")
        
        return speaker_files
    
    def transcribe_segments(self, segments, sr, labels, timestamps):
        """Transcribe audio segments by speaker."""
        print("Transcribing audio segments by speaker...")
        
        # Group segments by speaker
        speaker_segments = {}
        speaker_timestamps = {}
        for i, label in enumerate(labels):
            if label not in speaker_segments:
                speaker_segments[label] = []
                speaker_timestamps[label] = []
            speaker_segments[label].append(segments[i])
            speaker_timestamps[label].append(timestamps[i])
        
        # Transcribe each speaker's segments
        transcriptions = {}
        for speaker, segs in speaker_segments.items():
            transcriptions[speaker] = []
            
            for i, segment in enumerate(segs):
                try:
                    if self.transcriber:
                        # Convert audio to float32 if needed
                        segment = segment.astype(np.float32) if segment.dtype != np.float32 else segment
                        
                        # Actual transcription using the model
                        result = self.transcriber({"array": segment, "sampling_rate": sr})
                        text = result["text"]
                        
                        # Add timestamp
                        timestamp = speaker_timestamps[speaker][i]
                        time_str = f"[{timestamp:.2f}s]"
                        
                        transcriptions[speaker].append(f"{time_str} {text}")
                    else:
                        # Simulated transcription with timestamp
                        timestamp = speaker_timestamps[speaker][i]
                        time_str = f"[{timestamp:.2f}s]"
                        text = f"Transcribed text for Speaker {speaker+1}, segment {i+1}"
                        transcriptions[speaker].append(f"{time_str} {text}")
                except Exception as e:
                    print(f"Error transcribing segment {i} for speaker {speaker}: {e}")
                    transcriptions[speaker].append(f"[Error transcribing segment {i}]")
        
        return transcriptions
    
    def process_audio(self, audio_path, output_dir="output", n_speakers=2):
        """
        Process an audio file to separate speakers and transcribe their speech.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save the transcriptions
            n_speakers: Number of speakers to identify
        """
        print(f"Processing audio file: {audio_path}")
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"Error: Audio file '{audio_path}' not found.")
            return None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load audio file
        print("Loading audio file...")
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            print(f"Audio loaded: {len(audio)/sr:.2f} seconds at {sr}Hz")
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None
        
        # Segment the audio
        segments, timestamps, segment_length = self.segment_audio(audio, sr)
        print(f"Audio segmented into {len(segments)} segments of {segment_length}s each")
        
        if len(segments) == 0:
            print("Error: No complete segments found in audio file.")
            return None
        
        # Extract features for each segment
        features = []
        for segment in segments:
            segment_features = self.extract_features(segment, sr)
            # Use the mean of features across time
            features.append(np.mean(segment_features, axis=0))
        
        features = np.array(features)
        
        # Cluster segments by speaker
        labels = self.cluster_speakers(features, n_speakers)
        
        # Save audio by speaker
        speaker_files = self.save_audio_by_speaker(segments, labels, sr, output_dir)
        
        # Visualize the clustering
        print("Generating speaker separation visualization...")
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
        print(f"Visualization saved to {viz_file}")
        
        # Transcribe segments by speaker
        transcriptions = self.transcribe_segments(segments, sr, labels, timestamps)
        
        # Save transcriptions to files
        for speaker, texts in transcriptions.items():
            speaker_file = os.path.join(output_dir, f"speaker{speaker+1}_transcription.txt")
            
            print(f"Saving Speaker {speaker+1} transcription to {speaker_file}")
            with open(speaker_file, "w") as f:
                f.write("\n".join(texts))
        
        # Print transcriptions
        for speaker, texts in transcriptions.items():
            print(f"\nSpeaker {speaker+1} Transcription:")
            print("\n".join(texts[:5]) + (f"\n... [{len(texts)-5} more segments]" if len(texts) > 5 else ""))
        
        return {
            "transcriptions": transcriptions,
            "speaker_files": speaker_files,
            "visualization": viz_file
        }

# Example usage
if __name__ == "__main__":
    # Parse command line arguments if needed
    import argparse
    
    parser = argparse.ArgumentParser(description="Separate and transcribe voices in an audio file")
    parser.add_argument("--file", type=str, help="Path to the audio file")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--speakers", type=int, default=2, help="Number of speakers to identify")
    parser.add_argument("--model", type=str, default="openai/whisper-small", 
                        help="ASR model to use (default: openai/whisper-small)")
    
    args = parser.parse_args()
    
    # Initialize the transcriber
    transcriber = VoiceSeparationTranscriber(model_name=args.model)
    
    if args.file:
        # Process the specified audio file
        result = transcriber.process_audio(args.file, args.output, args.speakers)
        if result:
            print("\nProcessing complete!")
            print(f"Results saved to {args.output}/")
    else:
        print("\nNo audio file specified. Please provide a file path using --file.")
        print("Example usage: python voice_separator.py --file conversation.wav --speakers 2")