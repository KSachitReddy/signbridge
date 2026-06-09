import { useEffect, useRef, useState } from 'react';
import { useSignLanguageAI } from '../hooks/useSignLanguageAI';
import { estimateCustomGesture } from '../hooks/useFingerpose';

const CameraTranslationPage = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [prediction, setPrediction] = useState<string>('');
  const { isReady, recognize } = useSignLanguageAI();
  const requestRef = useRef<number | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (video) {
          video.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    };
    startCamera();

    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      if (video && video.srcObject) {
        (video.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (isReady && video) {
      const detectionLoop = () => {
        if (video) {
          const result = recognize(video);
          
          let finalPrediction = result?.gesture || 'Scanning...';
          
          // Check for custom gesture if landmarks are present
          if (result?.landmarks) {
            const custom = estimateCustomGesture(result.landmarks);
            if (custom) finalPrediction = custom;
          }
          
          setPrediction(finalPrediction);
        }
        requestRef.current = requestAnimationFrame(detectionLoop);
      };
      
      requestRef.current = requestAnimationFrame(detectionLoop);
      return () => {
        if (requestRef.current) cancelAnimationFrame(requestRef.current);
      };
    }
  }, [isReady, recognize]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-background p-4">
      <h1 className="text-3xl font-bold text-primary mb-4">Real-Time Camera Translation</h1>
      {!isReady && <p className="text-secondary mb-4 animate-pulse">Initializing AI Model...</p>}
      <video ref={videoRef} autoPlay playsInline className="w-full max-w-lg rounded-xl shadow-lg border-4 border-primary" />
      <div className="mt-6 p-4 bg-white rounded-lg shadow-md w-full max-w-lg text-center">
        <h2 className="text-lg font-semibold text-text">Detected Sign:</h2>
        <p className="text-2xl font-bold text-primary">{prediction || 'Scanning...'}</p>
      </div>
    </div>
  );
};

export default CameraTranslationPage;
