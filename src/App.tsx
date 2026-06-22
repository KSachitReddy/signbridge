import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import './App.css';
import { useVideoStreaming } from './useVideoStreaming';
import { AnalyticsDashboard } from './AnalyticsDashboard';
import { LanguageSwitcher } from './components/LanguageSwitcher';

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // Index
  [9, 10], [10, 11], [11, 12],   // Middle
  [13, 14], [14, 15], [15, 16], // Ring
  [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [5, 9], [9, 13], [13, 17] // Palm
];

function App() {
  const { t, i18n } = useTranslation();
  const { 
    videoRef, 
    canvasRef, 
    recognitionResult, 
    isClientMode, 
    modelsLoaded, 
    loadingText, 
    enrollFace 
  } = useVideoStreaming(i18n.language);

  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  
  const [enrollName, setEnrollName] = useState('');
  const [enrollStatus, setEnrollStatus] = useState('');
  const [emotionLogs, setEmotionLogs] = useState<any[]>([
    { emotion: 'Happy' },
    { emotion: 'Neutral' },
    { emotion: 'Neutral' },
  ]);

  // Handle Face Enrollment
  const handleEnroll = () => {
    if (!enrollName.trim()) {
      setEnrollStatus('Please enter a valid name.');
      return;
    }
    const success = enrollFace(enrollName.trim());
    if (success) {
      setEnrollStatus(`Registered face fingerprint for "${enrollName}"!`);
      setEnrollName('');
      setTimeout(() => setEnrollStatus(''), 4000);
    } else {
      setEnrollStatus('Failed. Please align your face in the camera view.');
    }
  };

  // Tracking emotion changes for the graph logs
  useEffect(() => {
    if (recognitionResult?.emotion?.emotion) {
      setEmotionLogs((prev) => {
        const next = [...prev, { emotion: recognitionResult.emotion.emotion }];
        // Limit to last 20 logs
        return next.slice(-20);
      });
    }
  }, [recognitionResult]);

  // Real-time canvas overlays (neon landmarks + face boxes)
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, 640, 480);

    if (!recognitionResult) return;

    // 1. Draw Hand Landmarks Connections & Points
    if (recognitionResult.gesture?.landmarks) {
      recognitionResult.gesture.landmarks.forEach((hand: any[]) => {
        // Draw bones
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.7)';
        ctx.lineWidth = 2.5;
        HAND_CONNECTIONS.forEach(([start, end]) => {
          if (hand[start] && hand[end]) {
            ctx.beginPath();
            ctx.moveTo(hand[start].x * 640, hand[start].y * 480);
            ctx.lineTo(hand[end].x * 640, hand[end].y * 480);
            ctx.stroke();
          }
        });

        // Draw joint points
        ctx.fillStyle = '#10b981'; // Neon emerald green
        hand.forEach((lm: any) => {
          ctx.beginPath();
          ctx.arc(lm.x * 640, lm.y * 480, 4.5, 0, 2 * Math.PI);
          ctx.fill();
        });
      });
    }

    // 2. Draw Face Bounding Box & Label
    if (recognitionResult.face?.results) {
      recognitionResult.face.results.forEach((face: any) => {
        if (face.box) {
          const [x, y, w, h] = face.box;
          
          // Draw Neon Purple bounding box
          ctx.strokeStyle = '#d946ef'; 
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, w, h);

          // Draw label background tag
          ctx.fillStyle = 'rgba(217, 70, 239, 0.9)';
          ctx.font = 'bold 14px sans-serif';
          const label = `${face.identity} (${Math.round((face.confidence || 0) * 100)}%)`;
          const textWidth = ctx.measureText(label).width;
          ctx.fillRect(x - 1.5, y - 24, textWidth + 14, 24);

          // Draw label text
          ctx.fillStyle = '#ffffff';
          ctx.fillText(label, x + 6, y - 7);
        }
      });
    }
  }, [recognitionResult]);

  const handleCopy = () => {
    if (recognitionResult?.gesture?.translated_text) {
      navigator.clipboard.writeText(recognitionResult.gesture.translated_text);
      alert('Copied translation to clipboard!');
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-left">
          <button className="nav-btn" onClick={() => window.location.reload()}>🏠</button>
          <h1>{t('title')}</h1>
        </div>
        <div className="header-right">
          <div className={`status-badge ${isClientMode ? 'client' : 'backend'}`}>
            {isClientMode ? (
              <span>⚡ Edge Mode (WASM {modelsLoaded ? 'Active' : 'Loading'})</span>
            ) : (
              <span>🟢 Socket Backend</span>
            )}
          </div>
          <LanguageSwitcher />
        </div>
      </header>

      {loadingText && (
        <div className="loading-overlay">
          <div className="loader"></div>
          <p>{loadingText}</p>
        </div>
      )}

      <main className="content">
        <div className="content-left">
          <div className="video-card glass">
            <video ref={videoRef} className="video-feed" muted playsInline />
            <canvas ref={canvasRef} style={{ display: 'none' }} width="640" height="480" />
            <canvas ref={overlayCanvasRef} className="overlay-canvas" width="640" height="480" />

            {recognitionResult && (
              <div className="overlay">
                <div className="overlay-item">
                  <span className="label-icon">👤</span>
                  <p>
                    <strong>{t('person')}:</strong> {recognitionResult.face?.results?.[0]?.identity || t('unknown')}
                  </p>
                </div>
                <div className="overlay-item">
                  <span className="label-icon">🖐️</span>
                  <p>
                    <strong>{t('gesture')}:</strong>{' '}
                    <span className="highlight-text">
                      {recognitionResult.gesture?.translated_text ||
                        recognitionResult.gesture?.label ||
                        t('none')}
                    </span>
                  </p>
                </div>
                <div className="overlay-item">
                  <span className="label-icon">😊</span>
                  <p>
                    <strong>{t('emotion')}:</strong> {recognitionResult.emotion?.emotion || t('neutral')}
                  </p>
                </div>
                <div className="overlay-meta">
                  <span>🌐 {i18n.language.toUpperCase()}</span>
                  <button className="copy-btn" onClick={handleCopy} title="Copy Translation">
                    📋 Copy
                  </button>
                </div>
              </div>
            )}
          </div>

          <div className="enrollment-card glass">
            <h3>👤 Known Face Registration</h3>
            <p>Type your name, look at the camera, and enroll to test known vs unknown face recognition.</p>
            <div className="enroll-form">
              <input 
                type="text" 
                placeholder="Enter your name..." 
                value={enrollName} 
                onChange={(e) => setEnrollName(e.target.value)} 
              />
              <button onClick={handleEnroll}>Register Face</button>
            </div>
            {enrollStatus && <p className="enroll-status">{enrollStatus}</p>}
          </div>
        </div>

        <div className="content-right">
          <AnalyticsDashboard logs={emotionLogs} />
          
          <div className="instructions-card glass">
            <h3>📖 Hackathon Guide</h3>
            <p>Perform one of these ISL gestures in view of the camera to see instant translation output:</p>
            <ul>
              <li><strong>👍 Thumbs Up</strong> — Agreement / Success</li>
              <li><strong>👎 Thumbs Down</strong> — Disagreement</li>
              <li><strong>🖐️ Open Palm</strong> — Stop / Hello</li>
              <li><strong>👈 Point Left</strong> — Left Indicator</li>
              <li><strong>👉 Point Right</strong> — Right Indicator</li>
              <li><strong>👋 Wave Hand</strong> — Hello / Greeting</li>
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
