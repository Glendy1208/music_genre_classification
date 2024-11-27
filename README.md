# Music Genre Classification

This project is a Flask-based web application that classifies the genre of a music file (MP3 or WAV) uploaded by the user. The application uses neural natwork backpropagation model for music genre classification. Model  trained by <a href='https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/code' target="_blank">GTZAN Dataset - Music Genre Classification</a>.

## Overview

The application allows users to:
1. Upload a music file in MP3 or WAV format.
2. Automatically convert MP3 files to WAV format using **FFmpeg**.
3. Extract features from the uploaded audio file using **Librosa**.
4. Predict the genre of the music using neural network backpropagation model.
5. Display the prediction result and allow users to listen to the uploaded file directly from the web interface.

The system also ensures efficient memory management by automatically deleting the uploaded files after 30 seconds.

## Features

- Supports MP3 to WAV conversion using FFmpeg.
- Predicts music genres such as **blues**, **classical**, **country**, **disco**, **hiphop**, **jazz**, **metal**, **pop**, **reggae**, **rock**.
- Automatically cleans up uploaded files after use in 1 hour.

---

## Installation Guide

### Prerequisites

Ensure that the following are installed on your system:
- Python 3.11 or higher
- FFmpeg (for audio file conversion)
- Virtual Environment (optional but recommended)

### Steps

1. **Clone the Repository** 
    - first create a folder for this repository clone. 
    - after that, run this command in terminal : git clone https://github.com/Glendy1208/music_genre_classification.git
2. **Install FFmpeg**
    - if you use windows, you can follow this tutorial for installation  FFmpeg : https://youtu.be/JR36oH35Fgg?si=1wFsMaJXY2rrhyYS
3. **Install Library**
    Run these comannd in terminal :
    - pip install flask
    - pip install librosa
    - pip install tensorflow
4. **Run Program**
    - in terminal run : python app.py
