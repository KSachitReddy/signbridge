import { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';

const BACKEND_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const socket = io(BACKEND_URL);

export const useVideoStreaming = (lang: string = 'en') => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [recognitionResult, setRecognitionResult] = useState<any>(null);
  const langRef = useRef(lang);

  // Keep language reference in sync
  useEffect(() => {
    langRef.current = lang;
  }, [lang]);

  // Set up socket listeners once
  useEffect(() => {
    const onConnect = () => console.log('Connected to backend API');
    const onResult = (data: any) => setRecognitionResult(data);

    socket.on('connect', onConnect);
    socket.on('recognition_result', onResult);

    return () => {
      socket.off('connect', onConnect);
      socket.off('recognition_result', onResult);
    };
  }, []);

  // Set up camera and frame emission loop once
  useEffect(() => {
    let intervalId: any = null;
    let activeStream: MediaStream | null = null;

    const startStream = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        activeStream = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch((err) => console.error("Video play error:", err));
        }

        intervalId = setInterval(() => {
          const canvas = canvasRef.current;
          if (canvas && videoRef.current) {
            const context = canvas.getContext('2d');
            if (context) {
              context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
              const frame = canvas.toDataURL('image/jpeg');
              socket.emit('frame', { frame, lang: langRef.current });
            }
          }
        }, 100); // 10 FPS
      } catch (error) {
        console.error("Camera access error:", error);
      }
    };

    startStream();

    return () => {
      if (intervalId) clearInterval(intervalId);
      if (activeStream) {
        activeStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return { videoRef, canvasRef, recognitionResult };
};
