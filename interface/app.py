import requests
import streamlit as st
import base64
import json
import librosa
import numpy as np
from pathlib import Path

# Set page config to wide mode to allow more horizontal space
st.set_page_config(layout="wide")

# Local API URL
API_URL = "http://localhost:8000/predict"

# # GCP API URL INACTIVE DUE TO BILLING
# API_URL = "https://instrument_classification-719648460452.europe-west2.run.app/predict"


# Convert image file to base64 for display
def convert_image_to_base64(file_path):
    image_path = Path(file_path)
    if image_path.exists():
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    return ""

# Load instrument data (for images and labels)
def load_instruments(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

# Detect the tempo from the .wav file using Librosa
def detect_tempo(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    if isinstance(tempo, (np.ndarray, list)):
        tempo = tempo[0]
    return tempo

# Display predictions
def display_predictions(predicted_instruments, instrument_data, tempo, rotate):
    if not predicted_instruments:
        st.write("No instruments predicted.")
        return

    # Ensure tempo is a valid number
    if tempo is None or tempo <= 0:
        tempo = 60  # Default to 60 BPM if invalid

    # Get the current flip state
    is_flipped = st.session_state.get('all_flipped', False)

    # Define the transform style based on flip state
    transform_style = "rotateY(180deg)" if is_flipped else "rotateY(0deg)"

    # Always use a single row with consistent card size
    num_instruments = len(predicted_instruments)

    # Create a single row of columns
    cols = st.columns(num_instruments)

    # Use consistent card size regardless of instrument count
    card_width = 238
    card_height = 438
    font_size = 24

    for col, instrument_info in zip(cols, predicted_instruments):
        instrument_name = instrument_info.get("instrument")
        probability = instrument_info.get("probability", 0)
        # Find instrument details in the JSON data
        instrument_data_item = next((item for item in instrument_data if item["instrument"] == instrument_name), None)
        if instrument_data_item:
            file_path = instrument_data_item["file_path"]
            file_path_v2 = instrument_data_item["file_path_v2"]
            base64_front = convert_image_to_base64(file_path)
            base64_back = convert_image_to_base64(file_path_v2)

            with col:
                # Create different hover behavior based on flipped state
                hover_css = """
                    .flip-card:hover .flip-card-inner {
                        transform: rotateY(0deg) !important;
                    }
                """ if is_flipped else """
                    .flip-card:hover .flip-card-inner {
                        transform: rotateY(180deg) !important;
                    }
                """

                st.markdown(
                    f"""
                    <style>
                        .flip-card {{
                            background-color: transparent;
                            width: {card_width}px;
                            height: {card_height}px;
                            perspective: 1000px;
                            margin: 10px auto;
                            border-radius: 10px;
                        }}
                        .flip-card-inner {{
                            position: relative;
                            width: 100%;
                            height: 100%;
                            text-align: center;
                            transform-style: preserve-3d;
                            transition: transform 0.6s;
                            transform: {transform_style};
                        }}
                        {hover_css}
                        .flip-card-front, .flip-card-back {{
                            position: absolute;
                            width: 100%;
                            height: 100%;
                            backface-visibility: hidden;
                            border-radius: 10px;
                            border: 3px solid #31333f;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                        }}
                        .flip-card-front {{
                            background-color: #51BE85;
                        }}
                        .flip-card-back {{
                            background-color: #67d2e4;
                            transform: rotateY(180deg);
                        }}
                        .flip-card img {{
                            width: 100%;
                            height: 100%;
                            object-fit: cover;
                            border-radius: 10px;
                        }}
                        .instrument-info {{
                            margin-top: 8px;
                            font-size: {font_size}px;
                            font-weight: bold;
                            text-align: center;
                            line-height: 1.3;
                            margin-bottom: 15px;
                        }}
                    </style>
                    <div class="flip-card">
                        <div class="flip-card-inner">
                            <div class="flip-card-front">
                                <img src="data:image/png;base64,{base64_front}" alt="Instrument Image Front" />
                            </div>
                            <div class="flip-card-back">
                                <img src="data:image/png;base64,{base64_back}" alt="Instrument Image Back" />
                            </div>
                        </div>
                    </div>
                    <div class="instrument-info">
                        {instrument_name}<br>
                        (prob: {probability:.1f}%)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# Main Streamlit app
def main():
    # Define and apply custom CSS for styling
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400&display=swap');

    html, body, [class*="css"] {
        font-family: 'Lexend', sans-serif;
    }

    .stButton > button {
        font-size: 1.5em;
        padding: 10px 20px;
    }

    .stButton.animate-play-pause > button {
        margin-top: 10px;
    }

    .stApp { background-color: #FEFFEF; }
    .stFileUploader > label { border-radius: 10px; padding: 10px; }
    .stButton > button {
        display: block;
        margin: 30px auto;
        background-color: #ffe433;
        font-size: 1.5em !important;
        padding: 15px 30px;
    }

    .custom-text {
        text-align: center;
        font-size: 1.5em;
        margin-bottom: 1em;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


    # Display headers
    st.markdown("<h1 style='text-align: center;'>OrchestrAIte</h1>", unsafe_allow_html=True)
    st.markdown('<div class="custom-text">Experience Classical Instrument Identification</div>', unsafe_allow_html=True)

    # Load instrument data
    instrument_data = load_instruments("interface/instruments.json")

    # Initialize session state variables
    session_state_defaults = {
        'is_playing': False,
        'prediction_made': False,
        'prediction': [],
        'tempo': None,
        'current_file': None,
        'all_flipped': False, # New single state for all flipped cards
    }
    for key, value in session_state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

    # Clear previous prediction if a new file is uploaded
    if uploaded_file is not None and uploaded_file != st.session_state.current_file:
        st.session_state.prediction_made = False
        st.session_state.prediction = []
        st.session_state.tempo = None
        st.session_state.current_file = uploaded_file
        st.session_state.all_flipped = False

    if uploaded_file is not None:
        st.audio(uploaded_file)

        # Create placeholder for prediction button that will disappear after use
        prediction_button_placeholder = st.empty()

        # Show Prediction button only if no prediction has been made yet
        if not st.session_state.prediction_made:
            with prediction_button_placeholder:
                # IMPORTANT: Explicitly show the Prediction button
                prediction_button = st.button("Identify Instruments", key="prediction_button")

                # Handle Prediction button click
                if prediction_button:
                    with st.spinner("Wait for it..."):
                        st.session_state.tempo = detect_tempo(uploaded_file)
                        try:
                            response = requests.post(API_URL, files={"file": uploaded_file.getvalue()})
                            if response.status_code == 200:
                                st.session_state.prediction = response.json().get("predictions", [])
                                st.session_state.prediction_made = True
                            else:
                                st.error("Error fetching predictions from the API")
                        except Exception as e:
                            st.error(f"Error with the prediction request: {e}")

                        # Clear the prediction button container after prediction is made
                        prediction_button_placeholder.empty()

                        # Force a rerun to display predictions immediately
                        st.rerun()

        # If predictions are available, show them
        if st.session_state.prediction_made:
            # Display predictions
            display_predictions(st.session_state.prediction, instrument_data, st.session_state.tempo, st.session_state.is_playing)

            # Add Flip All Cards button AFTER displaying predictions
            flip_button = st.button("Flip All", key="flip_all_btn")
            if flip_button:
                # Simply toggle the flipped state
                st.session_state.all_flipped = not st.session_state.all_flipped
                st.rerun()

if __name__ == "__main__":
    main()
