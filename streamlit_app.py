import streamlit as st
import torch
import torchaudio
import gdown
import os
import numpy as np
import matplotlib.pyplot as plt
from torchaudio.transforms import Resample, MFCC
from pydub import AudioSegment  # New import for MP3 support
import io

# Configuration
SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 400
HOP_LENGTH = 160
MODEL_URL = "https://drive.google.com/uc?id=1FdVrAZqoQ2Xz0GBEzDWTnexqWoX-oh6j"
MODEL_PATH = "best_model_safestudent.pt"
SUPPORTED_FORMATS = ['wav', 'mp3', 'flac', 'ogg']  # Now truly supports MP3

# (Keep your AudioCNN class and other functions exactly the same until load_audio_file)

def load_audio_file(uploaded_file):
    try:
        audio_bytes = uploaded_file.read()
        file_format = uploaded_file.name.split('.')[-1].lower()
        
        if file_format == 'mp3':
            # Use pydub for MP3 specifically
            audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            sample_rate = audio.frame_rate
            # Convert to numpy array
            data = np.array(audio.get_array_of_samples())
            if audio.channels > 1:
                data = data.reshape(-1, audio.channels).mean(axis=1)
        else:
            # Use torchaudio for other formats
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
            data = waveform.numpy()[0]  # Convert to mono numpy array
            
        return torch.from_numpy(data).float().unsqueeze(0), sample_rate
    
    except Exception as e:
        st.error(f"Error loading audio: {str(e)}")
        st.error("Supported formats: WAV, MP3, FLAC, OGG")
        return None, None

# (Rest of your code remains identical)
