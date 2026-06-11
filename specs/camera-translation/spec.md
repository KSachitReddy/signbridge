# Feature Spec: Camera Translation

## Status
Proposed

## Overview
Camera Translation uses Mediapipe and a webcam feed to translate sign language gestures into text and audio in real-time.

## Technical Specs
- **Input**: Video stream from `navigator.mediaDevices.getUserMedia`.
- **AI Model**: MediaPipe Tasks Vision (`@mediapipe/tasks-vision`).
- **Output**: Gesture label (string) and hand landmark coordinates.

## UI Mockup
- Camera View panel with green circular overlay showing hand landmarks.
- Controls: Speak, Copy, Emergency buttons.
