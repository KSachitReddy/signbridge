import { useState, useEffect, useRef } from 'react';
import {
  FilesetResolver,
  GestureRecognizer,
  type GestureRecognizerResult,
  type Landmark,
} from '@mediapipe/tasks-vision';

export const useSignLanguageAI = () => {
  const [isReady, setIsReady] = useState(false);
  const recognizerRef = useRef<GestureRecognizer | null>(null);

  useEffect(() => {
    const setup = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );
        recognizerRef.current = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
        });
        setIsReady(true);
      } catch (err) {
        console.error('Error initializing MediaPipe:', err);
      }
    };
    setup();

    return () => {
      if (recognizerRef.current) {
        recognizerRef.current.close();
      }
    };
  }, []);

  const recognize = (
    videoElement: HTMLVideoElement
  ): { gesture: string | null; landmarks: Landmark[] | null } => {
    if (!recognizerRef.current || videoElement.readyState !== 4)
      return { gesture: null, landmarks: null };

    const results: GestureRecognizerResult = recognizerRef.current.recognizeForVideo(
      videoElement,
      performance.now()
    );

    let gesture = null;
    let landmarks = null;
    if (results.gestures.length > 0) {
      gesture = results.gestures[0][0].categoryName;
      landmarks = results.landmarks[0]; // Extract 21 landmarks
    }
    return { gesture, landmarks };
  };

  return { isReady, recognize };
};
