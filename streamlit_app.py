import streamlit as st
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from PIL import Image
import io
import soundfile as sf
import gdown
import os
import matplotlib.pyplot as plt

# Configuration
MODEL_URL = "https://drive.google.com/uc?id=1Zvm-s-E4MbqCdeCCjG-6ggOaWzDJw-Jf"
MODEL_PATH = "GSC_DeFix.pt"
CLASSES = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 
    'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]

# Initialize model
class DeFixMatchModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(DeFixMatchModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, (8, 8))
        x = torch.flatten(x, start_dim=1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading model from Google Drive...'):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    
    model = DeFixMatchModel(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def transform_audio(waveform, sample_rate):
    transform = T.MelSpectrogram(
        sample_rate=16000,
        n_mels=64,
        n_fft=400,
        hop_length=160,
    )
    spectrogram = transform(waveform)
    
    target_length = 101
    if spectrogram.shape[-1] < target_length:
        pad_amount = target_length - spectrogram.shape[-1]
        spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_amount))
    elif spectrogram.shape[-1] > target_length:
        spectrogram = spectrogram[:, :, :target_length]
    
    return spectrogram

def plot_waveform(waveform, sample_rate):
    plt.figure(figsize=(10, 3))
    plt.plot(waveform.numpy().T)
    plt.title("Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

def plot_spectrogram(spec, title="Mel Spectrogram"):
    # Fix: Squeeze out the channel dimension and transpose
    spec_to_plot = spec.squeeze(0).numpy()  # Now shape (64, 101)
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_to_plot, cmap='viridis', aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.ylabel("Mel Bin")
    plt.xlabel("Time Frame")
    st.pyplot(plt)

# Streamlit UI
st.title("🎤 Speech Command Recognition")
st.write("Upload a 1-second audio clip to classify the speech command")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an audio file (WAV, MP3, FLAC...)",
    type=["wav", "mp3", "flac", "ogg"]
)

if uploaded_file is not None:
    try:
        audio_bytes = uploaded_file.read()
        
        with io.BytesIO(audio_bytes) as f:
            try:
                data, sample_rate = sf.read(f)
            except Exception as e:
                st.error(f"Could not read audio file: {str(e)}")
                st.error("Supported formats: WAV, MP3, FLAC, OGG")
                st.stop()
        
        if len(data.shape) > 1:
            data = data.mean(axis=1)
            
        duration = len(data)/sample_rate
        if not (0.8 <= duration <= 1.5):
            st.warning(f"For best results, use 1-second audio. Current: {duration:.2f}s")
        
        if duration > 1.0:
            data = data[:int(sample_rate*1)]
            st.info(f"Trimmed audio to first 1 second (original: {duration:.2f}s)")
            
        waveform = torch.from_numpy(data).float().unsqueeze(0)
        
        if sample_rate != 16000:
            resampler = T.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
            st.info(f"Resampled audio to 16kHz (original: {sample_rate}Hz)")
            
        col1, col2 = st.columns(2)
        with col1:
            st.audio(audio_bytes, format='audio/wav')
        with col2:
            st.write(f"Duration: {len(data)/sample_rate:.2f}s")
            st.write(f"Sample Rate: {sample_rate}Hz")
        
        plot_waveform(waveform, sample_rate)
        spectrogram = transform_audio(waveform, sample_rate)
        plot_spectrogram(spectrogram)
        
        with st.spinner('Classifying...'):
            with torch.no_grad():
                output = model(spectrogram.unsqueeze(0))  # Add batch dimension
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, top_idx = torch.topk(probabilities, 5)
                
        st.subheader("🔍 Prediction Results")
        for i in range(5):
            st.progress(float(top_prob[i]), text=f"{CLASSES[top_idx[i]]}: {top_prob[i]*100:.2f}%")
            
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        st.error("Please ensure you've uploaded a valid audio file (1 second duration works best)")
