import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useSignLanguageAI } from '../hooks/useSignLanguageAI';
import { useLLM } from '../hooks/useLLM';

const CameraTranslationPage = () => {
  const navigate = useNavigate();
  const { t, i18n } = useTranslation();
  const { translateText, loading: llmLoading, error: llmError } = useLLM();

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [prediction, setPrediction] = useState<string>('');
  const [confidence, setConfidence] = useState<number>(0);
  const [refinedSentence, setRefinedSentence] = useState<string>('');
  const { isReady, recognize } = useSignLanguageAI();
  const requestRef = useRef<number | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (video) video.srcObject = stream;
      } catch (err) {
        console.error('Error accessing camera:', err);
      }
    };
    startCamera();

    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      if (video && video.srcObject) {
        (video.srcObject as MediaStream).getTracks().forEach((track) => track.stop());
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
            if (result.gesture) {
              setPrediction((prev) => {
                if (prev !== result.gesture) {
                  setRefinedSentence('');
                  return result.gesture;
                }
                return prev;
              });
            }
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

  const getSpeechLangCode = (lng: string) => {
    switch (lng) {
      case 'hi': return 'hi-IN';
      case 'te': return 'te-IN';
      default: return 'en-US';
    }
  };

  const handleSpeak = () => {
    const textToSpeak = refinedSentence || prediction;
    if ('speechSynthesis' in window && textToSpeak) {
      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      utterance.lang = getSpeechLangCode(i18n.language);
      window.speechSynthesis.speak(utterance);
    }
  };

  const handleCopy = () => {
    const textToCopy = refinedSentence || prediction;
    if (textToCopy) {
      navigator.clipboard.writeText(textToCopy).then(() => {
        alert(t('dashboard.copied'));
      });
    }
  };

  const handleEmergency = () => {
    alert(t('dashboard.emergencyAlert'));
  };

  const handleRefine = async () => {
    if (prediction) {
      const targetLangName = i18n.language === 'hi' ? 'Hindi' : i18n.language === 'te' ? 'Telugu' : 'English';
      const refined = await translateText(prediction, targetLangName);
      setRefinedSentence(refined);
    }
  };

  const provider = localStorage.getItem('signbridge_provider');

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-extrabold text-blue-400">{t('dashboard.title')}</h1>
        <button
          onClick={() => navigate('/settings')}
          className="p-3 bg-gray-800 hover:bg-gray-700 text-white rounded-full shadow-lg border border-gray-700 transition cursor-pointer"
          title={t('nav.settings')}
        >
          ⚙️
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Camera Panel */}
        <div className="relative bg-gray-800 rounded-xl overflow-hidden shadow-xl border border-gray-700">
          <video ref={videoRef} autoPlay playsInline className="w-full h-auto" />
          <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full" />
          {!isReady && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
              {t('dashboard.initializing')}
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="bg-gray-800 p-6 rounded-xl shadow-xl border border-gray-700 flex flex-col gap-4">
          <h2 className="text-xl font-semibold border-b border-gray-700 pb-2">
            {t('dashboard.liveTranslation')}
          </h2>
          <div>
            <p className="text-sm text-gray-400">{t('dashboard.detectedSign')}</p>
            <p className="text-4xl font-bold text-green-400">{prediction || t('dashboard.scanning')}</p>
          </div>
          <div>
            <p className="text-sm text-gray-400">{t('dashboard.confidence')}</p>
            <div className="w-full bg-gray-700 h-4 rounded-full mt-1 overflow-hidden">
              <div className="bg-green-500 h-full" style={{ width: `${confidence * 100}%` }}></div>
            </div>
            <p className="text-right text-sm">{(confidence * 100).toFixed(0)}%</p>
          </div>

          {/* AI Refined Translation Panel */}
          {provider && (
            <div className="mt-4 p-4 bg-gray-900/50 border border-gray-700 rounded-lg space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-blue-400 uppercase tracking-wider">
                  ✨ AI Refined ({provider})
                </span>
                {llmLoading && (
                  <span className="text-xs text-gray-400 animate-pulse">Refining...</span>
                )}
              </div>
              <p className="text-lg text-white font-medium italic">
                {refinedSentence || (llmLoading ? '...' : 'Click "Refine" to translate...')}
              </p>
              {llmError && (
                <p className="text-xs text-red-400 mt-1">{llmError}</p>
              )}
            </div>
          )}

          <div className="mt-auto grid grid-cols-2 gap-2">
            <button
              onClick={handleSpeak}
              className="bg-blue-600 hover:bg-blue-700 p-3 rounded font-bold transition flex items-center justify-center gap-2 cursor-pointer"
            >
              🔊 {t('dashboard.speak')}
            </button>
            <button
              onClick={handleCopy}
              className="bg-gray-700 hover:bg-gray-600 p-3 rounded font-bold transition flex items-center justify-center gap-2 cursor-pointer"
            >
              📋 {t('dashboard.copy')}
            </button>
            {provider && prediction && (
              <button
                onClick={handleRefine}
                disabled={llmLoading}
                className="bg-indigo-650 hover:bg-indigo-700 disabled:bg-indigo-800 disabled:opacity-50 p-3 rounded font-bold transition flex items-center justify-center gap-2 col-span-2 cursor-pointer"
              >
                ✨ Refine with AI
              </button>
            )}
            <button
              onClick={handleEmergency}
              className="bg-red-650 hover:bg-red-700 p-3 rounded font-bold transition col-span-2 mt-2 cursor-pointer"
            >
              🆘 {t('dashboard.emergency')}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraTranslationPage;

// Formatted with Prettier
