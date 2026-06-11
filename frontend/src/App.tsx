import './App.css';
import { useVideoStreaming } from './useVideoStreaming';
import { AnalyticsDashboard } from './AnalyticsDashboard';

function App() {
  const { videoRef, canvasRef, recognitionResult } = useVideoStreaming();
  // Placeholder logs for now
  const logs = [{emotion: 'Happy'}, {emotion: 'Neutral'}];

  return (
    <div className="app-container">
      <header className="header">
        <h1>SignBridge AI</h1>
      </header>
      
      <main className="content">
        <div className="video-card glass">
          <video ref={videoRef} className="video-feed" muted playsInline />
          <canvas ref={canvasRef} style={{ display: 'none' }} width="640" height="480" />
          
          {recognitionResult && (
            <div className="overlay">
              <p>Person: {recognitionResult.face?.results?.[0]?.identity || 'Unknown'}</p>
            </div>
          )}
        </div>
        <AnalyticsDashboard logs={logs} />
      </main>
    </div>
  );
}

export default App;
