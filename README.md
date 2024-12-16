# OrchestrAIte: Instrument Detector 🎶🤖

Welcome to **OrchestrAIte** - an AI-powered instrument detector designed to revolutionize the way computers understand and interpret music! 🎵✨

## How We Did It 🚀
In just **two weeks**, we embarked on a journey to create an intelligent system that listens to music and predicts the instruments involved. 
Here’s a peek into our process:

### Data Collection

We utilized **MusicNet**, a comprehensive dataset, and performed data augmentation to create over **24,000+ audio files** in WAV with metadata in CSV format. 📊🔊

### Data Processing

- **EDA & Pre-processing**: Conducted exploratory data analysis and data cleaning.
- **Input & Output**: 
  - Input: Log-mel spectrograms
  - Output: Instrument predictions 🎹❗️

### Modeling

**Custom CNN**: 
  - Built a custom Convolutional Neural Network for multi-label classification.
  - Trained on 24,000+ segmented audio files

### Front-end & Deployment

- **Interface**: Developed using FastAPI & Streamlit for a seamless user experience.
- **Deployment**: Dockerized applications for easy deployment and scalability.

## Under the Hood 🔍
- **Input**: Accepts raw audio files.
- **Processing**: Features are extracted using advanced techniques.
- **Classification**: Multi-label CNN model predicts the instrument(s).


## Project Overview 🛠️

- **Duration**: 2 weeks
- **Dataset**: MusicNet, augmented to 24,000+ audio files
- **Process**: 
  - Input: Log-mel spectrograms
  - Output: Instrument predictions

---
The project was demoed at the Le Wagon, Tokyo, "Demo Day" on December 6, 2024.
