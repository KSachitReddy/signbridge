import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import './App.css';
import { useVideoStreaming } from './useVideoStreaming';
import { AnalyticsDashboard } from './AnalyticsDashboard';

function App() {
  const { t, i18n } = useTranslation();
  const { videoRef, canvasRef, recognitionResult } = useVideoStreaming();
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  const changeLanguage = (lng: string) => {
    i18n.changeLanguage(lng);
  };

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
        <h1>{t('title')}</h1>
        <div className="lang-switcher">
          <button onClick={() => changeLanguage('en')}>EN</button>
          <button onClick={() => changeLanguage('te')}>TE</button>
          <button onClick={() => changeLanguage('hi')}>HI</button>
          <button onClick={() => changeLanguage('ta')}>TA</button>
          <button onClick={() => changeLanguage('kn')}>KN</button>
          <button onClick={() => changeLanguage('ml')}>ML</button>
          <button onClick={() => changeLanguage('tcy')}>TCY</button>
        </div>
      </header>
      
      <main className="content">
        <div className="video-card glass">
          <video ref={videoRef} className="video-feed" muted playsInline />
          <canvas ref={canvasRef} style={{ display: 'none' }} width="640" height="480" />
          <canvas ref={overlayCanvasRef} className="overlay-canvas" width="640" height="480" />
          
          {recognitionResult && (
            <div className="overlay">
              <p>{t('person')}: {recognitionResult.face?.results?.[0]?.identity || t('unknown')}</p>
              <p>{t('gesture')}: {recognitionResult.gesture?.label || t('none')}</p>
              <p>{t('emotion')}: {recognitionResult.emotion?.emotion || t('neutral')}</p>
            </div>
          )}
        </div>
        <AnalyticsDashboard logs={[{emotion: 'Happy'}, {emotion: 'Neutral'}]} />
      </main>
    </div>
  );
}

export default App;
