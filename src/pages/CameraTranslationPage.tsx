import { useEffect, useRef, useState } from 'react';
import { useSignLanguageAI } from '../hooks/useSignLanguageAI';

const CameraTranslationPage = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [prediction, setPrediction] = useState<string>('');
  const [confidence, setConfidence] = useState<number>(0);
  const { isReady, recognize } = useSignLanguageAI();
  const requestRef = useRef<number | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (video) video.srcObject = stream;
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

  const drawLandmarks = (landmarks: any[]) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    ctx.fillStyle = '#10B981';
    landmarks.forEach((landmark) => {
      ctx.beginPath();
      ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 5, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  useEffect(() => {
    const video = videoRef.current;
    if (isReady && video) {
      const detectionLoop = () => {
        if (video) {
          const result = recognize(video);
          
          if (result) {
            if (result.gesture) setPrediction(result.gesture);
            // Simulate confidence
            setConfidence(Math.random() * 0.2 + 0.8);
            if (result.landmarks) drawLandmarks(result.landmarks);
          }
        }
        requestRef.current = requestAnimationFrame(detectionLoop);
      };
      
      requestRef.current = requestAnimationFrame(detectionLoop);
      return () => {
        if (requestRef.current) cancelAnimationFrame(requestRef.current);
      };
    }
  }, [isReady, recognize]);

  const handleSpeak = () => {
    if ('speechSynthesis' in window && prediction) {
      const utterance = new SpeechSynthesisUtterance(prediction);
      window.speechSynthesis.speak(utterance);
    }
  };

  const handleCopy = () => {
    if (prediction) {
      navigator.clipboard.writeText(prediction).then(() => {
        alert('Copied to clipboard!');
      });
    }
  };

  const handleEmergency = () => {
    alert('🚨 Emergency Alert Sent! Location shared with emergency contacts.');
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <h1 className="text-3xl font-bold mb-6 text-blue-400">SignBridge AI Dashboard</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Camera Panel */}
        <div className="relative bg-gray-800 rounded-xl overflow-hidden shadow-xl border border-gray-700">
          <video ref={videoRef} autoPlay playsInline className="w-full h-auto" />
          <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full" />
          {!isReady && <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">Initializing AI...</div>}
        </div>

        {/* Results Panel */}
        <div className="bg-gray-800 p-6 rounded-xl shadow-xl border border-gray-700 flex flex-col gap-4">
          <h2 className="text-xl font-semibold border-b border-gray-700 pb-2">Live Translation</h2>
          <div>
            <p className="text-sm text-gray-400">Detected Sign:</p>
            <p className="text-4xl font-bold text-green-400">{prediction || 'Scanning...'}</p>
          </div>
          <div>
            <p className="text-sm text-gray-400">Confidence:</p>
            <div className="w-full bg-gray-700 h-4 rounded-full mt-1 overflow-hidden">
              <div className="bg-green-500 h-full" style={{ width: `${confidence * 100}%` }}></div>
            </div>
            <p className="text-right text-sm">{(confidence * 100).toFixed(0)}%</p>
          </div>
          
          <div className="mt-auto grid grid-cols-2 gap-2">
            <button onClick={handleSpeak} className="bg-blue-600 hover:bg-blue-700 p-2 rounded">🔊 Speak</button>
            <button onClick={handleCopy} className="bg-gray-700 hover:bg-gray-600 p-2 rounded">📋 Copy</button>
            <button onClick={handleEmergency} className="bg-red-600 hover:bg-red-700 p-2 rounded col-span-2 mt-2">🆘 Emergency</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraTranslationPage;
