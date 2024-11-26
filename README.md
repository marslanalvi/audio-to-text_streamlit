# Whisper Audio Transcription with Streamlit

This is a simple Streamlit app that allows users to upload an audio file and get the transcription using OpenAI's Whisper model. The app supports multiple audio formats, including `.mp3`, `.wav`, `.m4a`, and `.opus`. It converts the uploaded audio file into a `.wav` format (if necessary) and then uses Whisper to transcribe the audio to text. The transcription is returned in **English** by default.

## Features
- Upload audio files in `.mp3`, `.wav`, `.m4a`, or `.opus` formats.
- Whisper model transcribes the audio to English text.
- Supports automatic conversion from `opus` format to `wav` format.
- Easy-to-use web interface built with Streamlit.

## Requirements
To run the app locally or deploy it to Streamlit Cloud, you'll need to install the following dependencies:

- **Python 3.7+**
- **Streamlit**
- **Whisper** (OpenAI’s Whisper model)
- **PyDub** (for audio conversion)
- **FFmpeg** (for handling audio files, required by PyDub)

### Install Dependencies
Create a virtual environment (optional but recommended) and install the required dependencies by running the following commands:

```bash
# Clone this repository
git clone https://github.com/marslanalvi/audio-to-text_streamlit.git

# Navigate to the project folder
cd audio-to-text_steamlit

# Create a virtual environment (optional)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
