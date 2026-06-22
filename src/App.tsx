import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import './App.css';
import { useVideoStreaming } from './useVideoStreaming';
import { AnalyticsDashboard } from './AnalyticsDashboard';
import {
  PROVIDERS,
  getProviderConfig,
  loadAISettings,
  saveAISettings,
  isProviderConfigured,
  testProviderConnection,
  enhanceTranslation,
  type AISettings,
  type ProviderId,
  type ConnectionTestResult,
} from './aiProviders';

const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4], // Thumb
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8], // Index
  [9, 10],
  [10, 11],
  [11, 12], // Middle
  [13, 14],
  [14, 15],
  [15, 16], // Ring
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20], // Pinky
  [5, 9],
  [9, 13],
  [13, 17], // Palm
];

interface Log {
  id: string;
  person: string;
  timestamp: string;
  sign: string;
  text: string;
  lang: string;
  confidence: number;
  aiEnhanced?: boolean;
}

interface FaceProfile {
  name: string;
  fingerprint: [number, number, number];
  dateAdded?: string;
  notes?: string;
}

function App() {
  const { t, i18n } = useTranslation();

  // 1. Navigation & App States
  const [activePage, setActivePage] = useState('Home');
  const [theme, setTheme] = useState(
    localStorage.getItem('signbridge_theme') || 'Standard Dark Theme'
  );

  // 2. Local Database states (synchronized with localStorage)
  const [conversations, setConversations] = useState<Log[]>(() => {
    try {
      const raw = localStorage.getItem('signbridge_conversations');
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  });

  const [people, setPeople] = useState<FaceProfile[]>(() => {
    try {
      const raw = localStorage.getItem('signbridge_faces');
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  });

  const [enrollName, setEnrollName] = useState('');
  const [enrollNotes, setEnrollNotes] = useState('');
  const [enrollStatus, setEnrollStatus] = useState('');

  // People registry manage states
  const [editingPersonName, setEditingPersonName] = useState<Record<string, string>>({});
  const [notesRecord, setNotesRecord] = useState<Record<string, string>>({});

  // Conversations search filter
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLogs, setSelectedLogs] = useState<Record<string, boolean>>({});

  // Streaming Hook - camera only starts while the Live Translation page is mounted
  const {
    videoRef,
    canvasRef,
    recognitionResult,
    isClientMode,
    loadingText,
    cameraError,
    enrollFace,
  } = useVideoStreaming(i18n.language, activePage === 'Live Translation');

  // Ollama connectivity status (local-only check, never probed from a hosted deployment)
  type OllamaStatus = 'checking' | 'available' | 'not_running' | 'hosted_unavailable';
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatus>('checking');

  useEffect(() => {
    const hostname = window.location.hostname;
    const isLocalHost = hostname === 'localhost' || hostname === '127.0.0.1';

    if (!isLocalHost) {
      setOllamaStatus('hosted_unavailable');
      return;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1500);

    fetch('http://localhost:11434/api/tags', { signal: controller.signal })
      .then((res) => setOllamaStatus(res.ok ? 'available' : 'not_running'))
      .catch(() => setOllamaStatus('not_running'))
      .finally(() => clearTimeout(timeoutId));

    return () => {
      clearTimeout(timeoutId);
      controller.abort();
    };
  }, []);

  const OLLAMA_STATUS_DISPLAY: Record<OllamaStatus, { label: string; className: string }> = {
    checking: { label: '⏳ Checking...', className: 'yellow' },
    available: { label: '✅ Local Ollama Available', className: 'green' },
    not_running: { label: '❌ Local Ollama Not Running', className: 'red' },
    hosted_unavailable: { label: '🌐 Ollama Unavailable In Hosted Deployment', className: 'neutral' },
  };

  // BYOK AI provider settings - `aiSettings` is the saved/active config used for live
  // translation enhancement; `aiDraft` is the editable form state on the Settings page.
  const [aiSettings, setAiSettings] = useState<AISettings>(() => loadAISettings());
  const [aiDraft, setAiDraft] = useState<AISettings>(aiSettings);
  const [aiTesting, setAiTesting] = useState(false);
  const [aiTestResult, setAiTestResult] = useState<ConnectionTestResult | null>(null);
  const [aiSaved, setAiSaved] = useState(false);

  const handleAiProviderChange = (id: ProviderId) => {
    setAiDraft((prev) => ({ ...prev, provider: id, model: getProviderConfig(id).defaultModel }));
    setAiTestResult(null);
  };

  const handleAiSave = () => {
    saveAISettings(aiDraft);
    setAiSettings(aiDraft);
    setAiSaved(true);
    setTimeout(() => setAiSaved(false), 2500);
  };

  const handleAiTest = async () => {
    setAiTesting(true);
    setAiTestResult(null);
    const result = await testProviderConnection(aiDraft);
    setAiTestResult(result);
    setAiTesting(false);
  };

  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  // Emotion charts logs
  const [emotionLogs, setEmotionLogs] = useState<any[]>([
    { emotion: 'Happy' },
    { emotion: 'Neutral' },
    { emotion: 'Neutral' },
  ]);

  // Logging throttles for auto log generator
  const [lastLoggedSign, setLastLoggedSign] = useState('');
  const [lastLogTime, setLastLogTime] = useState(0);

  // 3. Theme application
  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove('standard-dark-theme', 'high-contrast-dark-theme', 'large-text-mode');

    if (theme === 'High Contrast Dark Theme') {
      root.classList.add('high-contrast-dark-theme');
    } else if (theme === 'Large Text Mode') {
      root.classList.add('large-text-mode');
    } else {
      root.classList.add('standard-dark-theme');
    }
  }, [theme]);

  // Update people list whenever face database in localStorage changes
  const refreshPeopleRegistry = () => {
    try {
      const raw = localStorage.getItem('signbridge_faces');
      setPeople(raw ? JSON.parse(raw) : []);
    } catch {
      setPeople([]);
    }
  };

  // 4. Handle Face Enrollment
  const handleEnroll = () => {
    if (!enrollName.trim()) {
      setEnrollStatus('Please enter a name.');
      return;
    }
    const success = enrollFace(enrollName.trim());
    if (success) {
      // Custom attributes metadata updates
      try {
        const raw = localStorage.getItem('signbridge_faces');
        const db: FaceProfile[] = raw ? JSON.parse(raw) : [];
        const index = db.findIndex((p) => p.name.toLowerCase() === enrollName.trim().toLowerCase());
        if (index !== -1) {
          db[index].dateAdded = new Date().toLocaleDateString();
          db[index].notes = enrollNotes.trim() || 'No notes added.';
          localStorage.setItem('signbridge_faces', JSON.stringify(db));
        }
      } catch (err) {
        console.error('Failed to append registry metadata:', err);
      }

      setEnrollStatus(`Registered face fingerprint for "${enrollName}"!`);
      setEnrollName('');
      setEnrollNotes('');
      refreshPeopleRegistry();
      setTimeout(() => setEnrollStatus(''), 4000);
    } else {
      setEnrollStatus('Failed. Please align your face in the camera view.');
    }
  };

  // 5. Automatic Log generation when gesture detected
  useEffect(() => {
    const label = recognitionResult?.gesture?.label;
    const translated = recognitionResult?.gesture?.translated_text;
    if (label && label !== 'None' && label !== lastLoggedSign) {
      const now = Date.now();
      if (now - lastLogTime > 2500) {
        const logId = 'C_' + Math.random().toString(36).substring(2, 9).toUpperCase();
        const newLog: Log = {
          id: logId,
          person: recognitionResult?.face?.results?.[0]?.identity || 'Unknown',
          timestamp: new Date().toLocaleString(),
          sign: label,
          text: translated || label,
          lang: i18n.language.toUpperCase(),
          confidence: recognitionResult?.face?.results?.[0]?.confidence || 0.85,
        };
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setConversations((prev) => {
          const next = [newLog, ...prev];
          localStorage.setItem('signbridge_conversations', JSON.stringify(next));
          return next;
        });
        setLastLoggedSign(label);
        setLastLogTime(now);

        // Ask the configured BYOK provider for a richer interpretation of this gesture.
        // Patches the log in place once it arrives; silently keeps the static text otherwise.
        if (isProviderConfigured(aiSettings)) {
          enhanceTranslation(aiSettings, label, i18n.language).then((enhanced) => {
            if (!enhanced) return;
            setConversations((prev) => {
              const next = prev.map((log) =>
                log.id === logId ? { ...log, text: enhanced, aiEnhanced: true } : log
              );
              localStorage.setItem('signbridge_conversations', JSON.stringify(next));
              return next;
            });
          });
        }
      }
    }
  }, [recognitionResult, lastLoggedSign, lastLogTime, i18n.language, aiSettings]);

  // Tracking emotion changes for the graph logs
  useEffect(() => {
    if (recognitionResult?.emotion?.emotion) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setEmotionLogs((prev) => {
        const next = [...prev, { emotion: recognitionResult.emotion.emotion }];
        return next.slice(-20);
      });
    }
  }, [recognitionResult]);

  // 6. Real-time canvas overlays (neon landmarks + face boxes)
  useEffect(() => {
    // Only run canvas drawing if active tab is Live Translation
    if (activePage !== 'Live Translation') return;

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
  }, [recognitionResult, activePage]);

  // Copy Action
  const handleCopy = () => {
    if (recognitionResult?.gesture?.translated_text) {
      navigator.clipboard.writeText(recognitionResult.gesture.translated_text);
      alert(t('live.copied'));
    }
  };

  // Speak Action (Speech Synthesis)
  const handleSpeak = () => {
    const textToSpeak = recognitionResult?.gesture?.translated_text;
    if ('speechSynthesis' in window && textToSpeak && textToSpeak !== 'None') {
      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      const getSpeechLangCode = (lng: string) => {
        switch (lng) {
          case 'hi':
            return 'hi-IN';
          case 'te':
            return 'te-IN';
          default:
            return 'en-US';
        }
      };
      utterance.lang = getSpeechLangCode(i18n.language);
      window.speechSynthesis.speak(utterance);
    }
  };

  // Emergency Alert Mockup
  const handleEmergency = () => {
    alert(t('live.emergencyAlert'));
  };

  // 7. Registry Profiles logic
  const handleRenamePerson = (oldName: string) => {
    const newName = editingPersonName[oldName];
    if (!newName || !newName.trim()) return;

    try {
      const raw = localStorage.getItem('signbridge_faces');
      const db: FaceProfile[] = raw ? JSON.parse(raw) : [];
      const updated = db.map((p) => {
        if (p.name.toLowerCase() === oldName.toLowerCase()) {
          return { ...p, name: newName.trim() };
        }
        return p;
      });
      localStorage.setItem('signbridge_faces', JSON.stringify(updated));
      setEditingPersonName((prev) => ({ ...prev, [oldName]: '' }));
      refreshPeopleRegistry();
      alert('Profile renamed successfully!');
    } catch (err) {
      console.error(err);
    }
  };

  const handleUpdateNotes = (name: string) => {
    const newNotes = notesRecord[name];
    if (newNotes === undefined) return;

    try {
      const raw = localStorage.getItem('signbridge_faces');
      const db: FaceProfile[] = raw ? JSON.parse(raw) : [];
      const updated = db.map((p) => {
        if (p.name.toLowerCase() === name.toLowerCase()) {
          return { ...p, notes: newNotes.trim() || 'No notes added.' };
        }
        return p;
      });
      localStorage.setItem('signbridge_faces', JSON.stringify(updated));
      refreshPeopleRegistry();
      alert('Notes updated!');
    } catch (err) {
      console.error(err);
    }
  };

  const handleDeletePerson = (name: string) => {
    if (!window.confirm(t('people.confirmDelete'))) return;
    try {
      const raw = localStorage.getItem('signbridge_faces');
      const db: FaceProfile[] = raw ? JSON.parse(raw) : [];
      const filtered = db.filter((p) => p.name.toLowerCase() !== name.toLowerCase());
      localStorage.setItem('signbridge_faces', JSON.stringify(filtered));
      refreshPeopleRegistry();
    } catch (err) {
      console.error(err);
    }
  };

  // Export Registry Profile backup JSON
  const handleExportBackup = () => {
    const raw = localStorage.getItem('signbridge_faces') || '[]';
    const blob = new Blob([raw], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'signbridge_face_registry_backup.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Import Registry Profile backup JSON
  const handleImportBackup = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const parsed = JSON.parse(content);
        if (Array.isArray(parsed)) {
          localStorage.setItem('signbridge_faces', JSON.stringify(parsed));
          refreshPeopleRegistry();
          alert('Database registry successfully imported and restored!');
        } else {
          alert('Invalid backup file structure.');
        }
      } catch (err) {
        alert('Failed to parse backup JSON: ' + err);
      }
    };
    reader.readAsText(file);
  };

  // 8. Conversations log management
  const handleDeleteSelectedLogs = () => {
    if (!window.confirm(t('conversations.confirmDelete'))) return;
    const remaining = conversations.filter((log) => !selectedLogs[log.id]);
    setConversations(remaining);
    localStorage.setItem('signbridge_conversations', JSON.stringify(remaining));
    setSelectedLogs({});
  };

  const handleClearAllLogs = () => {
    if (!window.confirm(t('conversations.confirmDelete'))) return;
    setConversations([]);
    localStorage.setItem('signbridge_conversations', JSON.stringify([]));
  };

  const handleExportSelectedLogs = () => {
    const selectedList = conversations.filter((log) => selectedLogs[log.id]);
    const listToExport = selectedList.length > 0 ? selectedList : conversations;
    const blob = new Blob([JSON.stringify(listToExport, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'signbridge_conversation_history.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Filter conversations matching search query
  const filteredConversations = conversations.filter((log) => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return (
      log.person.toLowerCase().includes(q) ||
      log.sign.toLowerCase().includes(q) ||
      log.text.toLowerCase().includes(q)
    );
  });

  // 9. Side Bar menu settings
  const navItems = [
    { id: 'Home', label: t('nav.home'), icon: '🏠' },
    { id: 'Live Translation', label: t('nav.live'), icon: '📹' },
    { id: 'Conversations', label: t('nav.conversations'), icon: '💬' },
    { id: 'People', label: t('nav.people'), icon: '👥' },
    { id: 'Settings', label: t('nav.settings'), icon: '⚙️' },
    { id: 'About', label: t('nav.about'), icon: 'ℹ️' },
  ];

  // Helper to fetch translated page header titles
  const getPageHeader = () => {
    switch (activePage) {
      case 'Home':
        return t('home.title');
      case 'Live Translation':
        return t('live.cameraTitle');
      case 'Conversations':
        return t('conversations.title');
      case 'People':
        return t('people.title');
      case 'Settings':
        return t('settings.title');
      case 'About':
        return t('about.title');
      default:
        return 'SignBridge AI';
    }
  };

  // 10. Page View rendering dispatcher
  const renderPageContent = () => {
    switch (activePage) {
      case 'Home':
        return (
          <div className="home-container animate-fade-in">
            {/* Health Cards */}
            <div className="health-grid">
              <div className="glass-card status-card">
                <span className="card-icon">📷</span>
                <h5>Browser Camera</h5>
                <span className="status-label green">✅ Available</span>
              </div>
              <div className="glass-card status-card">
                <span className="card-icon">🧠</span>
                <h5>Ollama LLM</h5>
                <span className={`status-label ${OLLAMA_STATUS_DISPLAY[ollamaStatus].className}`}>
                  {OLLAMA_STATUS_DISPLAY[ollamaStatus].label}
                </span>
              </div>
              <div className="glass-card status-card">
                <span className="card-icon">💬</span>
                <h5>Conversations</h5>
                <span className="status-label count">{conversations.length}</span>
              </div>
              <div className="glass-card status-card">
                <span className="card-icon">👥</span>
                <h5>People</h5>
                <span className="status-label count purple">{people.length}</span>
              </div>
            </div>

            {/* Mission Statement */}
            <div className="glass-card mission-card">
              <h4>🌟 {t('about.missionHeader')}</h4>
              <p>{t('about.missionText')}</p>
            </div>

            {/* How It Works pipeline */}
            <h3 className="section-title">⚙️ How It Works (Pipeline)</h3>
            <div className="pipeline-grid">
              <div className="glass-card pipeline-card">
                <span className="pipeline-step-icon">📷</span>
                <h5>1. {t('home.sign')}</h5>
                <p>{t('home.sign_desc')}</p>
              </div>
              <div className="glass-card pipeline-card">
                <span className="pipeline-step-icon">✨</span>
                <h5>2. Landmark Tracking</h5>
                <p>Extracts face mesh landmarks, 21 hand joints, and poses in real-time.</p>
              </div>
              <div className="glass-card pipeline-card">
                <span className="pipeline-step-icon">🧠</span>
                <h5>3. Gesture Classifier</h5>
                <p>A temporal classification model detects sign sequences with high accuracy.</p>
              </div>
              <div className="glass-card pipeline-card">
                <span className="pipeline-step-icon">🔊</span>
                <h5>4. {t('home.voice')}</h5>
                <p>{t('home.voice_desc')}</p>
              </div>
            </div>

            {/* Supported Features Grid */}
            <div className="features-columns">
              <div className="glass-card list-card">
                <h4>🤟 Supported Gestures</h4>
                <div className="tags-flex">
                  <span className="tag-item pink">👍 Thumbs Up</span>
                  <span className="tag-item pink">👎 Thumbs Down</span>
                  <span className="tag-item pink">👈 Point Left</span>
                  <span className="tag-item pink">👉 Point Right</span>
                  <span className="tag-item pink">👆 Point Up</span>
                  <span className="tag-item pink">👇 Point Down</span>
                  <span className="tag-item pink">✋ Open Palm</span>
                  <span className="tag-item pink">👋 Hello</span>
                </div>
              </div>

              <div className="glass-card list-card">
                <h4>🌍 Supported Indian Languages</h4>
                <ul className="lang-list">
                  <li>
                    🇮🇳 <b>English</b> — default output
                  </li>
                  <li>
                    🇮🇳 <b>Hindi (हिंदी)</b> — translation & speech support
                  </li>
                  <li>
                    🇮🇳 <b>Telugu (తెలుగు)</b> — translation & speech support
                  </li>
                  <li>
                    🇮🇳 <b>Tamil (தமிழ்)</b> — translation support
                  </li>
                  <li>
                    🇮🇳 <b>Kannada (ಕನ್ನಡ)</b> — translation support
                  </li>
                  <li>
                    🇮🇳 <b>Malayalam (മലയാളം)</b> — translation support
                  </li>
                  <li>
                    🇮🇳 <b>Tulu (ತುಳು)</b> — translation support
                  </li>
                </ul>
              </div>
            </div>

            <div className="hero-action-row">
              <button onClick={() => setActivePage('Live Translation')} className="hero-start-btn">
                🚀 {t('home.btnStart')}
              </button>
            </div>
          </div>
        );

      case 'Live Translation':
        return (
          <div className="live-container animate-fade-in">
            <div className="content-left">
              {/* Webcam Card */}
              <div className="video-card glass">
                <video ref={videoRef} className="video-feed" muted playsInline />
                <canvas ref={canvasRef} style={{ display: 'none' }} width="640" height="480" />
                <canvas
                  ref={overlayCanvasRef}
                  className="overlay-canvas"
                  width="640"
                  height="480"
                />

                {cameraError && (
                  <div className="camera-error-banner">
                    <span className="camera-error-icon">⚠️</span>
                    <p>{cameraError}</p>
                  </div>
                )}

                {recognitionResult && (
                  <div className="overlay">
                    <div className="overlay-item">
                      <span className="label-icon">👤</span>
                      <p>
                        <strong>{t('live.person')}:</strong>{' '}
                        {recognitionResult.face?.results?.[0]?.identity || t('unknown')}
                      </p>
                    </div>
                    <div className="overlay-item">
                      <span className="label-icon">🖐️</span>
                      <p>
                        <strong>{t('live.sign')}:</strong>{' '}
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
                        <strong>{t('live.expression')}:</strong>{' '}
                        {recognitionResult.emotion?.emotion || t('neutral')}
                      </p>
                    </div>
                    <div className="overlay-meta">
                      <span>🌐 {i18n.language.toUpperCase()}</span>
                      <button className="copy-btn" onClick={handleCopy} title={t('live.btnCopy')}>
                        📋 {t('live.btnCopy')}
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Enrollment form widget */}
              <div className="enrollment-card glass">
                <h3>👤 {t('live.savePersonTitle')}</h3>
                <p>
                  Type details, look at the camera, and enroll to record your face fingerprint in
                  the local database registry.
                </p>
                <div className="enroll-form flex flex-col gap-3 mt-3">
                  <input
                    type="text"
                    placeholder={t('live.nameInput') + '...'}
                    value={enrollName}
                    onChange={(e) => setEnrollName(e.target.value)}
                    className="enroll-input"
                  />
                  <input
                    type="text"
                    placeholder={t('live.notesInput') + '...'}
                    value={enrollNotes}
                    onChange={(e) => setEnrollNotes(e.target.value)}
                    className="enroll-input"
                  />
                  <button onClick={handleEnroll} className="enroll-btn">
                    {t('live.btnSave')}
                  </button>
                </div>
                {enrollStatus && <p className="enroll-status">{enrollStatus}</p>}
              </div>
            </div>

            <div className="content-right">
              {/* Analytics graph log logs */}
              <AnalyticsDashboard logs={emotionLogs} />

              {/* Guidelines instructions */}
              <div className="instructions-card glass">
                <h3>📖 Hackathon Guide</h3>
                <p>
                  Perform one of these ISL gestures in view of the camera to see instant translation
                  output:
                </p>
                <ul className="guide-bullets">
                  <li>
                    <strong>👍 Thumbs Up</strong> — Agreement / Success
                  </li>
                  <li>
                    <strong>👎 Thumbs Down</strong> — Disagreement
                  </li>
                  <li>
                    <strong>🖐️ Open Palm</strong> — Stop / Hello
                  </li>
                  <li>
                    <strong>👈 Point Left</strong> — Left Indicator
                  </li>
                  <li>
                    <strong>👉 Point Right</strong> — Right Indicator
                  </li>
                  <li>
                    <strong>👋 Wave Hand</strong> — Hello / Greeting
                  </li>
                </ul>

                <div className="action-button-group border-t border-gray-700 pt-4 mt-4 flex flex-col gap-2">
                  <button onClick={handleSpeak} className="action-btn-blue">
                    🔊 {t('live.btnSpeak')}
                  </button>
                  <button onClick={handleEmergency} className="action-btn-red">
                    🆘 {t('live.btnEmergency')}
                  </button>
                </div>
              </div>
            </div>
          </div>
        );

      case 'Conversations':
        return (
          <div className="conversations-container glass p-6 animate-fade-in">
            <div className="conversations-header flex flex-wrap justify-between items-center gap-4 mb-6">
              <div className="flex items-center gap-2 bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 w-full sm:w-80">
                <span>🔍</span>
                <input
                  type="text"
                  placeholder={t('conversations.filterPerson') + '...'}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="bg-transparent border-none text-white outline-none w-full text-sm"
                />
              </div>
              <div className="flex gap-2 w-full sm:w-auto">
                <button
                  onClick={handleDeleteSelectedLogs}
                  className="btn-action-red flex-1 sm:flex-initial text-sm py-2 px-3"
                >
                  🗑️ {t('conversations.btnDeleteSelected')}
                </button>
                <button
                  onClick={handleClearAllLogs}
                  className="btn-action-red flex-1 sm:flex-initial text-sm py-2 px-3"
                >
                  💥 {t('conversations.btnDeleteAll')}
                </button>
                <button
                  onClick={handleExportSelectedLogs}
                  className="btn-action-blue flex-1 sm:flex-initial text-sm py-2 px-3"
                >
                  📥 {t('conversations.btnExport')}
                </button>
              </div>
            </div>

            {filteredConversations.length === 0 ? (
              <div className="empty-state text-center py-10 text-gray-400">
                {t('conversations.emptyLogs')}
              </div>
            ) : (
              <div className="table-responsive">
                <table className="logs-table w-full text-left text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="p-3 w-8">
                        <input
                          type="checkbox"
                          onChange={(e) => {
                            const checked = e.target.checked;
                            const next: Record<string, boolean> = {};
                            filteredConversations.forEach((log) => {
                              next[log.id] = checked;
                            });
                            setSelectedLogs(next);
                          }}
                          checked={
                            filteredConversations.length > 0 &&
                            filteredConversations.every((log) => selectedLogs[log.id])
                          }
                        />
                      </th>
                      <th className="p-3">{t('conversations.cols.id')}</th>
                      <th className="p-3">{t('conversations.cols.person')}</th>
                      <th className="p-3">{t('conversations.cols.timestamp')}</th>
                      <th className="p-3">{t('conversations.cols.sign')}</th>
                      <th className="p-3">{t('conversations.cols.text')}</th>
                      <th className="p-3">{t('conversations.cols.lang')}</th>
                      <th className="p-3">{t('conversations.cols.confidence')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredConversations.map((log) => (
                      <tr key={log.id} className="border-b border-gray-800 hover:bg-gray-800/40">
                        <td className="p-3">
                          <input
                            type="checkbox"
                            checked={!!selectedLogs[log.id]}
                            onChange={(e) => {
                              setSelectedLogs((prev) => ({ ...prev, [log.id]: e.target.checked }));
                            }}
                          />
                        </td>
                        <td className="p-3 font-mono text-xs text-blue-400">{log.id}</td>
                        <td className="p-3 font-semibold">{log.person}</td>
                        <td className="p-3 text-gray-400 text-xs">{log.timestamp}</td>
                        <td className="p-3">
                          <span className="badge-sign">{log.sign}</span>
                        </td>
                        <td className="p-3 text-green-400 font-semibold">
                          {log.text}
                          {log.aiEnhanced && (
                            <span className="ai-enhanced-badge" title="Refined by your configured AI provider">
                              ✨
                            </span>
                          )}
                        </td>
                        <td className="p-3 font-bold text-gray-400">{log.lang}</td>
                        <td className="p-3 text-xs">{(log.confidence * 100).toFixed(0)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        );

      case 'People':
        return (
          <div className="people-container animate-fade-in">
            {people.length === 0 ? (
              <div className="glass-card p-6 text-center text-gray-400">
                No saved people profiles in the database registry yet. You can enroll faces during
                Live Translation.
              </div>
            ) : (
              <div>
                <h3 className="section-title">👤 Registry Profiles</h3>
                <div className="people-grid">
                  {people.map((p) => (
                    <div key={p.name} className="glass-card person-card">
                      <div className="card-top-accent"></div>
                      <h4 className="card-name">👤 {p.name}</h4>
                      <div className="card-details text-xs text-gray-400">
                        <p>
                          <strong>{t('people.dateAdded')}:</strong>{' '}
                          {p.dateAdded || new Date().toLocaleDateString()}
                        </p>
                        <p>
                          <strong>Notes:</strong> {p.notes || 'None'}
                        </p>
                      </div>

                      <div className="card-actions mt-4 pt-3 border-t border-gray-800 flex flex-col gap-2">
                        <div className="flex gap-2">
                          <input
                            type="text"
                            placeholder="New name..."
                            value={editingPersonName[p.name] || ''}
                            onChange={(e) =>
                              setEditingPersonName((prev) => ({
                                ...prev,
                                [p.name]: e.target.value,
                              }))
                            }
                            className="rename-input flex-1 p-1 bg-gray-900 border border-gray-700 rounded text-xs text-white"
                          />
                          <button
                            onClick={() => handleRenamePerson(p.name)}
                            className="btn-rename text-xs px-2 py-1 bg-blue-600 rounded text-white font-semibold"
                          >
                            ✏️
                          </button>
                        </div>

                        <div className="flex gap-2">
                          <input
                            type="text"
                            placeholder="Edit notes..."
                            value={notesRecord[p.name] || ''}
                            onChange={(e) =>
                              setNotesRecord((prev) => ({ ...prev, [p.name]: e.target.value }))
                            }
                            className="rename-input flex-1 p-1 bg-gray-900 border border-gray-700 rounded text-xs text-white"
                          />
                          <button
                            onClick={() => handleUpdateNotes(p.name)}
                            className="btn-rename text-xs px-2 py-1 bg-purple-600 rounded text-white font-semibold"
                          >
                            📝
                          </button>
                        </div>

                        <button
                          onClick={() => handleDeletePerson(p.name)}
                          className="btn-delete-person text-xs py-1 px-2 border border-red-650 text-red-400 rounded hover:bg-red-950/20"
                        >
                          🗑️ {t('people.btnDelete')}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Backup settings */}
            <div className="glass-card backup-card mt-8 p-6">
              <h4>📂 {t('people.importExport')}</h4>
              <p className="text-xs text-gray-400 mb-4">
                Export or import json registries to share databases across different client
                browsers.
              </p>

              <div className="backup-buttons-row">
                <button onClick={handleExportBackup} className="action-btn-blue text-sm">
                  📥 {t('people.btnExportDb')}
                </button>
                <div className="file-import-container">
                  <label
                    htmlFor="import-backup-file"
                    className="action-btn-purple text-sm inline-block cursor-pointer text-center"
                  >
                    📤 {t('people.btnImportDb')}
                  </label>
                  <input
                    type="file"
                    id="import-backup-file"
                    accept=".json"
                    onChange={handleImportBackup}
                    style={{ display: 'none' }}
                  />
                </div>
              </div>
            </div>
          </div>
        );

      case 'Settings':
        return (
          <div className="settings-container glass p-6 animate-fade-in max-w-xl mx-auto">
            <h3 className="text-lg font-bold border-b border-gray-700 pb-2 mb-6">
              ⚙️ {t('settings.title')}
            </h3>

            <div className="space-y-6">
              {/* Language selection */}
              <div className="flex flex-col gap-2">
                <label className="text-sm font-semibold text-gray-300">
                  {t('settings.language')}
                </label>
                <select
                  value={i18n.language}
                  onChange={(e) => i18n.changeLanguage(e.target.value)}
                  className="p-2.5 rounded-lg bg-gray-900 border border-gray-700 text-white w-full text-sm outline-none"
                >
                  <option value="en">English</option>
                  <option value="hi">हिन्दी (Hindi)</option>
                  <option value="te">తెలుగు (Telugu)</option>
                  <option value="ta">தமிழ் (Tamil)</option>
                  <option value="kn">ಕನ್ನಡ (Kannada)</option>
                  <option value="ml">മലയാളം (Malayalam)</option>
                  <option value="tcy">ತುಳು (Tulu)</option>
                </select>
              </div>

              {/* Theme selection */}
              <div className="flex flex-col gap-2">
                <label className="text-sm font-semibold text-gray-300">{t('settings.theme')}</label>
                <select
                  value={theme}
                  onChange={(e) => {
                    setTheme(e.target.value);
                    localStorage.setItem('signbridge_theme', e.target.value);
                  }}
                  className="p-2.5 rounded-lg bg-gray-900 border border-gray-700 text-white w-full text-sm outline-none"
                >
                  <option value="Standard Dark Theme">Standard Dark Theme</option>
                  <option value="High Contrast Dark Theme">High Contrast Dark Theme</option>
                  <option value="Large Text Mode">Large Text Mode</option>
                </select>
              </div>

              {/* AI providers selection (BYOK) */}
              <div className="flex flex-col gap-3 pt-4 border-t border-gray-800">
                <h4 className="text-sm font-bold text-blue-400">✨ AI LLM BYOK Settings</h4>
                <p className="text-xs text-gray-400">
                  Bring your own API key to enhance live translations with AI-assisted
                  interpretation. Keys are stored only in this browser's local storage and sent
                  directly to the provider.
                </p>

                <div className="flex flex-col gap-2">
                  <label className="text-xs font-semibold text-gray-300">Provider</label>
                  <select
                    value={aiDraft.provider}
                    onChange={(e) => handleAiProviderChange(e.target.value as ProviderId)}
                    className="p-2.5 rounded-lg bg-gray-900 border border-gray-700 text-white w-full text-sm outline-none"
                  >
                    {PROVIDERS.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.label}
                      </option>
                    ))}
                  </select>
                </div>

                {getProviderConfig(aiDraft.provider).requiresApiKey && (
                  <div className="flex flex-col gap-2">
                    <label className="text-xs font-semibold text-gray-300">API Key</label>
                    <input
                      type="password"
                      value={aiDraft.apiKey}
                      onChange={(e) => setAiDraft((prev) => ({ ...prev, apiKey: e.target.value }))}
                      placeholder={getProviderConfig(aiDraft.provider).keyPlaceholder}
                      autoComplete="off"
                      className="p-2.5 rounded-lg bg-gray-900 border border-gray-700 text-white w-full text-sm outline-none"
                    />
                  </div>
                )}

                {aiDraft.provider === 'ollama' && (
                  <div className="flex flex-col gap-2">
                    <label className="text-xs font-semibold text-gray-300">Ollama Base URL</label>
                    <input
                      type="text"
                      value={aiDraft.baseUrl}
                      onChange={(e) => setAiDraft((prev) => ({ ...prev, baseUrl: e.target.value }))}
                      placeholder="http://localhost:11434"
                      className="p-2.5 rounded-lg bg-gray-900 border border-gray-700 text-white w-full text-sm outline-none"
                    />
                  </div>
                )}

                <div className="flex flex-col gap-2">
                  <label className="text-xs font-semibold text-gray-300">Model</label>
                  <input
                    type="text"
                    list="ai-model-suggestions"
                    value={aiDraft.model}
                    onChange={(e) => setAiDraft((prev) => ({ ...prev, model: e.target.value }))}
                    placeholder={getProviderConfig(aiDraft.provider).defaultModel}
                    className="p-2.5 rounded-lg bg-gray-900 border border-gray-700 text-white w-full text-sm outline-none"
                  />
                  <datalist id="ai-model-suggestions">
                    {getProviderConfig(aiDraft.provider).suggestedModels.map((m) => (
                      <option key={m} value={m} />
                    ))}
                  </datalist>
                </div>

                <div className="flex gap-2 pt-1">
                  <button
                    onClick={handleAiSave}
                    className="action-btn-blue text-sm flex-1 cursor-pointer"
                  >
                    {aiSaved ? '✅ Saved' : '💾 Save Settings'}
                  </button>
                  <button
                    onClick={handleAiTest}
                    disabled={aiTesting}
                    className="action-btn-purple text-sm flex-1 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {aiTesting ? '⏳ Testing...' : '🔌 Test Connection'}
                  </button>
                </div>

                {aiTestResult && (
                  <div className={`ai-test-result ${aiTestResult.ok ? 'ok' : 'fail'}`}>
                    {aiTestResult.ok ? '✅' : '❌'} {aiTestResult.message}
                  </div>
                )}

                <p className="ai-active-provider">
                  Active provider: <strong>{getProviderConfig(aiSettings.provider).label}</strong> (
                  {aiSettings.model})
                  {!isProviderConfigured(aiSettings) && ' — not configured, using local translation only.'}
                </p>
              </div>

              <div className="pt-4 border-t border-gray-800">
                <button
                  onClick={() => {
                    if (
                      window.confirm(
                        'Clear all local databases (registry faces & conversation logs)?'
                      )
                    ) {
                      localStorage.removeItem('signbridge_faces');
                      localStorage.removeItem('signbridge_conversations');
                      refreshPeopleRegistry();
                      setConversations([]);
                      alert('Databases successfully cleared.');
                    }
                  }}
                  className="w-full text-sm text-center border border-red-650 text-red-400 font-bold p-3 rounded-lg hover:bg-red-950/20 transition cursor-pointer"
                >
                  🗑️ {t('settings.dbMgmt')} (Clear Database)
                </button>
              </div>
            </div>
          </div>
        );

      case 'About':
        return (
          <div className="about-container glass p-6 animate-fade-in max-w-2xl mx-auto space-y-6">
            <div>
              <h3 className="text-lg font-bold border-b border-gray-700 pb-2 text-purple-400">
                ℹ️ {t('about.missionHeader')}
              </h3>
              <p className="text-sm leading-relaxed mt-3">{t('about.missionText')}</p>
            </div>

            <div className="pt-4 border-t border-gray-800">
              <h4 className="font-bold text-sm text-blue-400 mb-2">{t('about.shortcutsTitle')}</h4>
              <ul className="list-disc list-inside text-xs text-gray-400 space-y-1.5">
                <li>
                  Press{' '}
                  <kbd className="px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700">H</kbd>{' '}
                  to open Home dashboard.
                </li>
                <li>
                  Press{' '}
                  <kbd className="px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700">L</kbd>{' '}
                  to open Live Camera translation.
                </li>
                <li>
                  Press{' '}
                  <kbd className="px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700">C</kbd>{' '}
                  to view Conversation logs.
                </li>
                <li>
                  Press{' '}
                  <kbd className="px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700">P</kbd>{' '}
                  to manage enrolled People registry.
                </li>
                <li>
                  Press{' '}
                  <kbd className="px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700">S</kbd>{' '}
                  to configure Settings.
                </li>
              </ul>
            </div>

            <div className="pt-4 border-t border-gray-800">
              <h4 className="font-bold text-sm text-green-400 mb-2">
                {t('about.dictionaryTitle')}
              </h4>
              <p className="text-xs text-gray-400 mb-3">
                The local classifier supports translation for 30 Indian Sign Language gestures in
                real-time, including:
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                <span className="p-2 bg-gray-800/40 rounded border border-gray-800">
                  👍 Agreement / Yes
                </span>
                <span className="p-2 bg-gray-800/40 rounded border border-gray-800">
                  👎 Disagreement / No
                </span>
                <span className="p-2 bg-gray-800/40 rounded border border-gray-800">
                  👋 Hello / Welcome
                </span>
                <span className="p-2 bg-gray-800/40 rounded border border-gray-800">
                  🖐️ Stop / Wait
                </span>
                <span className="p-2 bg-gray-800/40 rounded border border-gray-800">
                  👈 Point Left
                </span>
                <span className="p-2 bg-gray-800/40 rounded border border-gray-800">
                  👉 Point Right
                </span>
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  // Keyboard shortcut listeners (nav.home, nav.live, etc.)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const active = document.activeElement;
      if (
        active &&
        (active.tagName === 'INPUT' ||
          active.tagName === 'TEXTAREA' ||
          active.closest('input') ||
          active.closest('textarea'))
      ) {
        return;
      }
      const key = e.key.toLowerCase();
      if (key === 'h') setActivePage('Home');
      else if (key === 'l') setActivePage('Live Translation');
      else if (key === 'c') setActivePage('Conversations');
      else if (key === 'p') setActivePage('People');
      else if (key === 's') setActivePage('Settings');
      else if (key === 'a') setActivePage('About');
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="app-container flex min-h-screen">
      {/* 1. Left Sidebar Navigation */}
      <aside className="sidebar flex flex-col justify-between p-5 border-r border-gray-700">
        <div className="flex flex-col gap-6">
          <div className="logo-header flex items-center gap-2">
            <span className="logo-icon text-3xl">🤟</span>
            <div className="logo-text">
              <h2 className="text-base font-extrabold text-white leading-none">SignBridge AI</h2>
              <span className="text-[11px] text-gray-400 tracking-wider uppercase font-semibold">
                ISL Platform
              </span>
            </div>
          </div>

          <nav className="nav-menu flex flex-col gap-1.5">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                className={`nav-item flex items-center gap-3 py-2 px-3.5 rounded-lg text-sm transition-all text-left ${activePage === item.id ? 'active' : ''}`}
              >
                <span className="text-base">{item.icon}</span>
                <span>{item.label}</span>
              </button>
            ))}
          </nav>
        </div>

        <div className="sidebar-bottom flex flex-col gap-4 border-t border-gray-800 pt-4">
          <div className="quick-lang-selector flex flex-col gap-1.5">
            <label className="text-[11px] text-gray-400 font-bold uppercase tracking-wider">
              {t('settings.language')}
            </label>
            <select
              value={i18n.language}
              onChange={(e) => i18n.changeLanguage(e.target.value)}
              className="lang-select text-sm p-2 rounded-lg bg-gray-900 border border-gray-700 text-white cursor-pointer w-full outline-none"
            >
              <option value="en">🌐 English</option>
              <option value="hi">🌐 Hindi (हिंदी)</option>
              <option value="te">🌐 Telugu (తెలుగు)</option>
              <option value="ta">🌐 Tamil (தமிழ்)</option>
              <option value="kn">🌐 Kannada (ಕನ್ನಡ)</option>
              <option value="ml">🌐 Malayalam (മലയാളം)</option>
              <option value="tcy">🌐 Tulu (ತುಳು)</option>
            </select>
          </div>

          <div
            className={`app-mode-badge text-center py-1.5 rounded-lg text-[11px] font-bold tracking-wider uppercase ${isClientMode ? 'client' : 'backend'}`}
          >
            {isClientMode ? '⚡ Edge Mode' : '🟢 Socket Backend'}
          </div>
        </div>
      </aside>

      {/* 2. Right Main Content Panel */}
      <main className="main-content flex-1 flex flex-col min-h-screen bg-gray-950 text-white">
        {/* Header */}
        <header className="header flex items-center justify-between p-4 border-b border-gray-800">
          <h2 className="text-xl font-bold">{getPageHeader()}</h2>
          <div className="header-right flex items-center gap-4">
            {activePage !== 'Live Translation' && (
              <button
                onClick={() => setActivePage('Live Translation')}
                className="action-btn-primary text-sm px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-bold transition flex items-center gap-2 cursor-pointer"
              >
                🚀 {t('home.btnStart')}
              </button>
            )}
          </div>
        </header>

        {/* Global Loading Overlay */}
        {loadingText && activePage === 'Live Translation' && (
          <div className="loading-overlay">
            <div className="loader"></div>
            <p className="mt-3 text-sm">{loadingText}</p>
          </div>
        )}

        {/* Dynamic page render */}
        <div className="page-wrapper flex-1 p-6 overflow-y-auto">{renderPageContent()}</div>
      </main>
    </div>
  );
}

export default App;
