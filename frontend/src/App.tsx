import { useEffect, useRef } from 'react';
import './App.css';
import { useVideoStreaming } from './useVideoStreaming';
import { AnalyticsDashboard } from './AnalyticsDashboard';

function App() {
  const { videoRef, canvasRef, recognitionResult } = useVideoStreaming();
  const overlayCanvasRef = useRef(null);

  useEffect(() => {
    if (recognitionResult && overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, 640, 480);

      // Draw Neon Hands
      if (recognitionResult.gesture?.landmarks) {
        ctx.strokeStyle = '#00ffff'; // Neon blue/cyan
        ctx.lineWidth = 3;
        recognitionResult.gesture.landmarks.forEach((hand) => {
          hand.forEach((lm) => {
            ctx.beginPath();
            ctx.arc(lm.x * 640, lm.y * 480, 5, 0, 2 * Math.PI);
            ctx.stroke();
          });
        });
      }
    }
  }, [recognitionResult]);

  return (
    <div className="app-container">
      <header className="header">
        <h1>SignBridge AI</h1>
      </header>

      <main className="content">
        <div className="video-card glass">
          <video ref={videoRef} className="video-feed" muted playsInline />
          <canvas ref={canvasRef} style={{ display: 'none' }} width="640" height="480" />
          <canvas ref={overlayCanvasRef} className="overlay-canvas" width="640" height="480" />

          {recognitionResult && (
            <div className="overlay">
              <p>Person: {recognitionResult.face?.results?.[0]?.identity || 'Unknown'}</p>
              <p>Gesture: {recognitionResult.gesture?.label || 'None'}</p>
              <p>Emotion: {recognitionResult.emotion?.emotion || 'Neutral'}</p>
            </div>
          )}
        </div>
        <AnalyticsDashboard logs={[{ emotion: 'Happy' }, { emotion: 'Neutral' }]} />
      </main>
    </div>
  );
}

export default App;
