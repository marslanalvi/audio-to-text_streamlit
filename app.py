import streamlit as st
import whisper
import tempfile
from pydub import AudioSegment
import io

# Load the Whisper model
model = whisper.load_model("small")


# Function to convert audio to WAV format
def convert_to_wav(audio_file):
    audio = AudioSegment.from_file(audio_file)  # Auto-detect format
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        audio.export(temp_wav, format="wav")
        temp_wav.close()
        return temp_wav.name


# Function to transcribe the audio file
def transcribe_audio(audio_file):
    # Convert the file to WAV format (if it's not already)
    if audio_file.name.endswith(".opus"):
        # Convert the audio to WAV if it's in Opus format
        audio_path = convert_to_wav(audio_file)
    else:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_audio:
            tmp_audio.write(audio_file.getvalue())  # Write the uploaded file to a temp file
            tmp_audio.close()
            audio_path = tmp_audio.name

    # Transcribe the audio file using Whisper (force English transcription)
    result = model.transcribe(audio_path, language="en")
    return result["text"]


# Streamlit app UI
st.title("Whisper Audio Transcription")
st.write("Upload an audio file and let Whisper transcribe it for you.")

# File uploader widget
audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "opus"])

if audio_file is not None:
    # Display the uploaded audio file
    st.audio(audio_file, format="audio/wav")

    # Transcribe the uploaded audio file
    with st.spinner("Transcribing..."):
        transcription = transcribe_audio(audio_file)

    # Display the transcription result
    st.subheader("Transcription Result:")
    st.write(transcription)
