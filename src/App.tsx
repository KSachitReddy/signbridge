import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import './App.css';
import { useVideoStreaming } from './useVideoStreaming';
import { AnalyticsDashboard } from './AnalyticsDashboard';
import { LanguageSwitcher } from './components/LanguageSwitcher';

function App() {
  const { t, i18n } = useTranslation();
  const { videoRef, canvasRef, recognitionResult } = useVideoStreaming(i18n.language);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (recognitionResult && overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, 640, 480);
        
        // Draw Neon Hands
        if (recognitionResult.gesture?.landmarks) {
          ctx.strokeStyle = '#00ffff'; // Neon blue/cyan
          ctx.lineWidth = 3;
          recognitionResult.gesture.landmarks.forEach((hand: any[]) => {
            hand.forEach((lm: any) => {
              ctx.beginPath();
              ctx.arc(lm.x * 640, lm.y * 480, 5, 0, 2 * Math.PI);
              ctx.stroke();
            });
          });
        }
      }
    }
  }, [recognitionResult]);

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-left">
          <button className="nav-btn">🏠</button>
          <h1>{t('title')}</h1>
        </div>
        <LanguageSwitcher />
      </header>
      
      <main className="content">
        <div className="video-card glass">
          <video ref={videoRef} className="video-feed" muted playsInline />
          <canvas ref={canvasRef} style={{ display: 'none' }} width="640" height="480" />
          <canvas ref={overlayCanvasRef} className="overlay-canvas" width="640" height="480" />
          
          {recognitionResult && (
            <div className="overlay">
              <p>{t('person')}: {recognitionResult.face?.results?.[0]?.identity || t('unknown')}</p>
              <p>{t('gesture')}: {recognitionResult.gesture?.translated_text || recognitionResult.gesture?.label || t('none')}</p>
              <p>{t('emotion')}: {recognitionResult.emotion?.emotion || t('neutral')}</p>
              <p>🌐 {i18n.language.toUpperCase()}</p>
              <button className="copy-btn" title="Copy Translation">📋</button>
            </div>
          )}
        </div>
        <AnalyticsDashboard logs={[
          {emotion: 'Happy'}, {emotion: 'Happy'}, 
          {emotion: 'Neutral'}, {emotion: 'Neutral'}, {emotion: 'Neutral'},
          {emotion: 'Sad'}
        ]} />
      </main>
    </div>
  );
}

export default App;
