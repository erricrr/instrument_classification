# OrchestrAIte: AI-Powered Instrument Identification

OrchestrAIte is an AI-powered system that analyzes `.wav` audio files and identifies the instruments being played. It leverages deep learning techniques to process music data efficiently.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model and Data](#model-and-data)
- [Deployment](#deployment)
- [Contributors](#contributors)

## Overview

OrchestrAIte processes raw `.wav` audio input, extracts features, and predicts multiple instruments using a convolutional neural network (CNN).

### Key Features
- Accepts `.wav` audio files as input
- Uses log-mel spectrograms for feature extraction
- Multi-label CNN for instrument identification
- Web interface built with FastAPI and Streamlit
- Deployable via Docker and Google Cloud Run

## Installation

### Clone the Repository
```sh
gh repo clone erricrr/instrument_classification
cd instrument_classification
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage

### Run Locally

1. Start the FastAPI backend:
   ```sh
   uvicorn api.fast_api:app --reload
   ```
2. Run the Streamlit interface in another terminal:
   ```sh
   python -m streamlit run interface/app.py
   ```

### Run with Docker

1. Set up a Google Cloud Project.
2. Build and push the Docker image.
3. Deploy the FastAPI backend to Cloud Run.
4. Update `API_URL` in `interface/app.py`.
5. Test the deployed API.

## Model and Data

**Dataset**: The training data comes from the [MusicNet dataset on Kaggle](https://www.kaggle.com/datasets/imsparsh/musicnet-dataset), which is pre-split into training and test folders. Although MusicNet contains labels for 11 instruments in the training set, only 7 instruments are labeled in the test set. As a result, the model was trained to identify the following instruments:

1. Piano
2. Violin
3. Viola
4. Cello
5. Bassoon
6. Clarinet
7. Horn

**Preprocessing**: Exploratory data analysis, data cleaning, and feature extraction

**Model**: Custom CNN for multi-label classification trained on 24,000+ samples

**Input**: `.wav` audio files converted into log-mel spectrograms

**Output**: Instrument identification

## Limitations

- The model is trained only on a subset of instruments from the MusicNet dataset, so it may not recognize all instruments in a given audio file.
- It may misidentify instruments, especially in complex polyphonic recordings.


## Contributors

OrchestrAIte was developed by a four-person team as part of a project at Le Wagon Tokyo. The project was completed in two weeks and demoed on December 6, 2024.
