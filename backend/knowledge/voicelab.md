# VoiceLab — Advanced Audio AI Command Center

Final-year project. Also described as "Emotion-Aware Voice Processing for Customer Services."

Talal's flagship, most important project.

## Tech stack
- Frontend: React, Vite, React Router, CSS Modules, and JavaScript
- Backend: Python and FastAPI, using a REST API
- AI and ML: Whisper for speech-to-text, TensorFlow, CNN-BiLSTM for emotion recognition, librosa, noisereduce, gTTS, and FFmpeg
- Database: none — VoiceLab does not use a database

## Features
- Noise Reduction — cleans background noise from audio input
- Voice Enhancement — improves audio quality and clarity
- Text-to-Speech — English, 4 voices, 2 male and 2 female
- Voice-to-Text — powered by OpenAI Whisper
- Emotion Detection — analyzes a voice recording and predicts speaker emotion using a CNN-BiLSTM model. The emotion-detection endpoint returns the emotion, a confidence score, and all scores across categories.
- Accepted formats for emotion detection: MP3 and WAV, max about 5 megabytes file size. An earlier duration limit of about 20 seconds is not confirmed for the current build — don't state it unless verified.

## Design
Dark, modern, AI-focused, dashboard-oriented interface. Key colors used: a near-black background, a dark slate panel color, and a bright cyan accent.

## Talal's role — important, state accurately
"I worked on the frontend architecture and integration of VoiceLab, while the project also includes a Python and FastAPI AI backend." Do not claim personal ownership of every backend component — the backend documentation and implementation were shared with a teammate.
