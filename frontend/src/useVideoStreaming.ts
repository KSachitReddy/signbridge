import { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';

const socket = io('http://localhost:8000');

export const useVideoStreaming = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [recognitionResult, setRecognitionResult] = useState(null);

  useEffect(() => {
    socket.on('connect', () => console.log('Connected to backend'));
    socket.on('recognition_result', (data) => setRecognitionResult(data));

    const startStream = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        
        // Stream frames
        setInterval(() => {
          const canvas = canvasRef.current;
          const context = canvas.getContext('2d');
          context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
          const frame = canvas.toDataURL('image/jpeg');
          socket.emit('frame', frame);
        }, 100); // 10 FPS
      }
    };
    startStream();
  }, []);

  return { videoRef, canvasRef, recognitionResult };
};
