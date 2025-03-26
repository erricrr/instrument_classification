from fastapi import FastAPI, UploadFile, File, HTTPException
from model_loader import ModelLoader
import tensorflow as tf
import librosa
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import os
from fastapi.responses import JSONResponse

app = FastAPI()

# Load the model when the API starts
model_loader = ModelLoader('./model/CNN_model1_sliced2.keras')
model = model_loader.load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    This endpoint handles the instrument identification.
    It receives audio files and returns predictions using the model.
    """

    unique_instruments = [1, 41, 42, 43, 61, 71, 72]

    mlb = MultiLabelBinarizer(classes=unique_instruments)
    mlb.fit([unique_instruments])  # Fit the binarizer

    try:
        audio_data = await file.read()
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(audio_data)
        predictions_with_probs = predict_instruments(temp_file, model, mlb)
        print(f"Predicted instruments for '{temp_file}': {predictions_with_probs}")

        INSTRUMENT_MAP = {
        1: 'Piano',
        41: 'Violin',
        42: 'Viola',
        43: 'Cello',
        61: 'Horn',
        71: 'Bassoon',
        72: 'Clarinet',
        }

        # Sort predictions by probability in descending order
        sorted_predictions = sorted(predictions_with_probs, key=lambda x: x[1], reverse=True)

        # Format predictions for API response
        instrument_predictions = [
            {
                "instrument": INSTRUMENT_MAP.get(inst_id, "Unknown"),
                "probability": round(prob * 100, 2)  # Convert to percentage
            }
            for inst_id, prob in sorted_predictions
        ]

        os.remove(temp_file)

        return JSONResponse(content={'predictions':instrument_predictions})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """
    This endpoint lets users verify the API is running and helps with debugging.
    """
    print(model)
    return {"message": "Instrument Classifier API is running"}


def extract_mel_spectrogram(file_path):
    """
    Extract Mel-spectrogram from the audio file.
    """
    y, sr = librosa.load(file_path, sr=None, duration=5.0)  # Limit to 5 seconds to keep data consistent
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibels for better visualization
    return mel_spec_db


def predict_instruments(file_path, model, mlb, thresh=0.45):
    """
    Predict instruments from the audio file and return the probabilities.
    """
    # Extract Mel-spectrogram from the input file
    mel_spec = extract_mel_spectrogram(file_path)

    # Reshape spectrogram to match model input shape
    mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add channel dimension
    mel_spec = tf.keras.preprocessing.sequence.pad_sequences([mel_spec],
                                                              padding="post",
                                                              dtype="float32",
                                                              value=0)

    # Predict the instruments and their probabilities
    prediction = model.predict(mel_spec)  # The shape of 'prediction' should be (1, num_classes)

    # Filter predictions by the threshold
    predictions_with_probs = [
        (mlb.classes_[idx], prob)
        for idx, prob in enumerate(prediction[0])
        if prob > thresh
    ]

    return predictions_with_probs
