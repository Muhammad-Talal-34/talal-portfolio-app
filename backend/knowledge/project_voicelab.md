# VoiceLab — Advanced Audio AI Command Center

Final-year project, 2025 to 2026, University of Education, Lahore. Also described as "Emotion-Aware Voice Processing for Customer Services."

Talal's flagship, most important academic project.

## Tech stack
- Frontend: React, Vite, React Router, CSS Modules, and JavaScript
- Backend: Python and FastAPI, using a REST API
- AI and ML: TensorFlow, CNN-BiLSTM for emotion recognition, Whisper (base model) for speech-to-text, Edge TTS for text-to-speech, librosa, noisereduce, pyloudnorm, and FFmpeg
- Database: none — VoiceLab does not use a database

## Features
- Emotion Detection — a CNN-BiLSTM model for 5-class speech emotion recognition (Angry, Fear, Happy, Neutral, Sad), using Mel spectrogram and MFCC features. Trained on the RAVDESS and TESS datasets. The endpoint returns the emotion, a confidence score, and all scores across categories.
- Voice-to-Text — powered by OpenAI Whisper
- Text-to-Speech — English, 4 voices (Lisa, Aria, Mike, David), using Edge TTS
- Noise Reduction — cleans background noise from audio input
- Voice Enhancement — improves audio quality and clarity
- Accepted formats for emotion detection: MP3 and WAV, max about 5 megabytes file size. An earlier duration limit of about 20 seconds is not confirmed for the current build — don't state it unless verified.
- Frontend: live browser recording, audio playback, and waveform visualization.

## Important accuracy notes
- Not real-time streaming: users can record in the browser, and the backend processes the complete recorded file. Don't call the emotion model "real-time."
- The model was trained on acted speech datasets, so results on noisy, natural call audio may be lower. No final accuracy figure is confirmed — don't state one.

## Design
Dark, modern, AI-focused, dashboard-oriented interface. Key colors used: a near-black background, a dark slate panel color, and a bright cyan accent.

## Talal's role
"I trained the CNN-BiLSTM emotion model, built the FastAPI REST endpoints that serve it, integrated Whisper, Edge TTS, noise reduction, and voice enhancement, and built the React interface." Describe it as his own final-year project. Don't add technical detail beyond what is written here.
