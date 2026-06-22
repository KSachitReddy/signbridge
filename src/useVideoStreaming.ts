import { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';
import { FilesetResolver, HandLandmarker, FaceLandmarker } from '@mediapipe/tasks-vision';

const BACKEND_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Translation map for client-side mode
const TRANSLATION_MAP: Record<string, Record<string, string>> = {
  hi: {
    none: 'कोई नहीं',
    hello: 'नमस्ते',
    'thank you': 'धन्यवाद',
    'thumbs up': 'अंगूठा ऊपर (बहुत बढ़िया)',
    'thumbs down': 'अंगूठा नीचे (असहमत)',
    'open palm': 'खुली हथेली',
    'point left': 'बाईं ओर इशारा',
    'point right': 'दाईं ओर इशारा',
    unknown: 'अज्ञात',
  },
  te: {
    none: 'ఏమీ లేదు',
    hello: 'నమస్కారం',
    'thank you': 'ధన్యవాదాలు',
    'thumbs up': 'అభినందనలు (థంబ్స్ అప్)',
    'thumbs down': 'అసమ్మతి (థంబ్స్ డౌన్)',
    'open palm': 'తెరచిన చేయి',
    'point left': 'ఎడమ వైపు చూపించు',
    'point right': 'కుడి వైపు చూపించు',
    unknown: 'తెలియదు',
  },
};

const getTranslatedText = (label: string, lang: string) => {
  const g = label.toLowerCase();
  const l = lang.toLowerCase().substring(0, 2);
  if (TRANSLATION_MAP[l] && TRANSLATION_MAP[l][g]) {
    return TRANSLATION_MAP[l][g];
  }
  return label;
};

// Simple heuristic gesture recognition
const classifyGestureClient = (landmarks: any[]): string => {
  if (!landmarks || landmarks.length < 21) return 'None';

  // Extract key points
  const thumbTip = landmarks[4];
  const thumbIP = landmarks[3];
  const thumbMCP = landmarks[2];

  const indexTip = landmarks[8];
  const indexPIP = landmarks[6];
  const indexKnuckle = landmarks[5];

  const middleTip = landmarks[12];
  const middlePIP = landmarks[10];
  const middleKnuckle = landmarks[9];

  const ringTip = landmarks[16];
  const ringPIP = landmarks[14];
  const ringKnuckle = landmarks[13];

  const pinkyTip = landmarks[20];
  const pinkyPIP = landmarks[18];
  const pinkyKnuckle = landmarks[17];

  // Helper to determine if a finger is extended
  const isIndexExtended = indexTip.y < indexPIP.y && indexPIP.y < indexKnuckle.y;
  const isMiddleExtended = middleTip.y < middlePIP.y && middlePIP.y < middleKnuckle.y;
  const isRingExtended = ringTip.y < ringPIP.y && ringPIP.y < ringKnuckle.y;
  const isPinkyExtended = pinkyTip.y < pinkyPIP.y && pinkyPIP.y < pinkyKnuckle.y;

  // Thumb extended detection (horizontally or vertically away from MCP)
  const isThumbExtended =
    Math.abs(thumbTip.x - indexKnuckle.x) > 0.08 || Math.abs(thumbTip.y - indexKnuckle.y) > 0.08;

  // 1. Thumbs Up
  // Thumb is up, all other fingers are folded (y of tips is lower than knuckles in screen space/greater value)
  if (
    thumbTip.y < thumbIP.y &&
    thumbIP.y < thumbMCP.y &&
    indexTip.y > indexKnuckle.y &&
    middleTip.y > middleKnuckle.y &&
    ringTip.y > ringKnuckle.y &&
    pinkyTip.y > pinkyKnuckle.y
  ) {
    return 'Thumbs Up';
  }

  // 2. Thumbs Down
  // Thumb is down, other fingers are folded
  if (
    thumbTip.y > thumbIP.y &&
    thumbIP.y > thumbMCP.y &&
    indexTip.y > indexKnuckle.y &&
    middleTip.y > middleKnuckle.y &&
    ringTip.y > ringKnuckle.y &&
    pinkyTip.y > pinkyKnuckle.y
  ) {
    return 'Thumbs Down';
  }

  // 3. Point Left
  // Index finger extended horizontally left, all other fingers closed
  if (
    indexTip.x < indexKnuckle.x &&
    Math.abs(indexTip.y - indexKnuckle.y) < Math.abs(indexTip.x - indexKnuckle.x) &&
    middleTip.x > indexKnuckle.x &&
    ringTip.x > indexKnuckle.x &&
    pinkyTip.x > indexKnuckle.x
  ) {
    return 'Point Left';
  }

  // 4. Point Right
  // Index finger extended horizontally right, all other fingers closed
  if (
    indexTip.x > indexKnuckle.x &&
    Math.abs(indexTip.y - indexKnuckle.y) < Math.abs(indexTip.x - indexKnuckle.x) &&
    middleTip.x < indexKnuckle.x &&
    ringTip.x < indexKnuckle.x &&
    pinkyTip.x < indexKnuckle.x
  ) {
    return 'Point Right';
  }

  // 5. Open Palm / Hello
  if (isIndexExtended && isMiddleExtended && isRingExtended && isPinkyExtended && isThumbExtended) {
    return 'Open Palm';
  }

  // Fallback / Hello/Wave detection (same as open palm with fingers up)
  if (isIndexExtended && isMiddleExtended && isRingExtended && isPinkyExtended) {
    return 'Hello';
  }

  return 'None';
};

// Persistent database in localStorage
interface FaceProfile {
  name: string;
  fingerprint: [number, number, number];
}

const getFaceDB = (): FaceProfile[] => {
  try {
    const raw = localStorage.getItem('signbridge_faces');
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
};

const saveFaceDB = (db: FaceProfile[]) => {
  localStorage.setItem('signbridge_faces', JSON.stringify(db));
};

export const useVideoStreaming = (lang: string = 'en') => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [recognitionResult, setRecognitionResult] = useState<any>(null);
  const [isClientMode, setIsClientMode] = useState<boolean>(false);
  const [modelsLoaded, setModelsLoaded] = useState<boolean>(false);
  const [loadingText, setLoadingText] = useState<string>('Connecting to backend...');
  const [activeFaceLandmarks, setActiveFaceLandmarks] = useState<any>(null);

  const langRef = useRef(lang);
  const socketRef = useRef<any>(null);

  // Keep language reference in sync
  useEffect(() => {
    langRef.current = lang;
  }, [lang]);

  // Connect Socket.IO and implement fallback timeout
  useEffect(() => {
    console.log('Attempting connection to backend at:', BACKEND_URL);
    const socket = io(BACKEND_URL, {
      timeout: 1500,
      reconnectionAttempts: 1,
    });
    socketRef.current = socket;

    const connectionTimeout = setTimeout(() => {
      if (!socket.connected) {
        console.warn('Socket connection timed out. Falling back to client-side MediaPipe.');
        setIsClientMode(true);
        setLoadingText('Initializing MediaPipe WASM on-device models...');
      }
    }, 1500);

    socket.on('connect', () => {
      clearTimeout(connectionTimeout);
      console.log('Connected to backend API successfully.');
      setIsClientMode(false);
      setLoadingText('');
    });

    socket.on('recognition_result', (data: any) => {
      setRecognitionResult(data);
    });

    socket.on('connect_error', () => {
      console.warn('Socket connection error. Activating client-side fallback.');
      setIsClientMode(true);
      setLoadingText('Initializing MediaPipe WASM on-device models...');
    });

    return () => {
      clearTimeout(connectionTimeout);
      socket.disconnect();
    };
  }, []);

  // Client-Side MediaPipe Engine
  useEffect(() => {
    if (!isClientMode) return;

    let faceLandmarker: FaceLandmarker | null = null;
    let handLandmarker: HandLandmarker | null = null;
    let animationFrameId: number | null = null;
    let activeStream: MediaStream | null = null;

    const initMediaPipe = async () => {
      try {
        setLoadingText('Loading Fileset Resolver...');
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
        );

        setLoadingText('Loading Face Detector task...');
        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/face_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          outputFaceBlendshapes: true,
        });

        setLoadingText('Loading Hand Detector task...');
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/hand_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: 2,
        });

        console.log('Client-side MediaPipe processors loaded successfully.');
        setModelsLoaded(true);
        setLoadingText('');
      } catch (err) {
        console.error('Failed to load local models, trying Google Storage CDN fallbacks...', err);
        try {
          const vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
          );
          faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            outputFaceBlendshapes: true,
          });

          handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numHands: 2,
          });
          setModelsLoaded(true);
          setLoadingText('');
        } catch (cdnErr) {
          console.error('All MediaPipe model initializations failed:', cdnErr);
          setLoadingText('Error loading models. Running in Demo Mode.');
        }
      }
    };

    const processFrame = () => {
      if (!videoRef.current || !canvasRef.current) {
        animationFrameId = requestAnimationFrame(processFrame);
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      if (ctx && video.readyState === video.HAVE_ENOUGH_DATA) {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const timestamp = performance.now();

        // 1. Process Face Recognition
        let faceResult = { status: 'success', results: [] as any[] };
        let faceLandmarksData: any = null;

        if (faceLandmarker) {
          const faceDetections = faceLandmarker.detectForVideo(video, timestamp);
          if (faceDetections.faceLandmarks && faceDetections.faceLandmarks.length > 0) {
            const landmarks = faceDetections.faceLandmarks[0];
            faceLandmarksData = landmarks; // Save reference for face enrollment

            // Calculate ratios for face recognition
            // Left eye outer corner: 33, Right eye outer corner: 263
            const p33 = landmarks[33];
            const p263 = landmarks[263];
            const eyeDist = Math.sqrt(
              Math.pow(p33.x - p263.x, 2) +
                Math.pow(p33.y - p263.y, 2) +
                Math.pow(p33.z - p263.z, 2)
            );

            // Nose tip: 4, Chin: 152
            const p4 = landmarks[4];
            const p152 = landmarks[152];
            const noseToChin = Math.sqrt(
              Math.pow(p4.x - p152.x, 2) + Math.pow(p4.y - p152.y, 2) + Math.pow(p4.z - p152.z, 2)
            );

            // Mouth corners: 61 and 291
            const p61 = landmarks[61];
            const p291 = landmarks[291];
            const mouthWidth = Math.sqrt(
              Math.pow(p61.x - p291.x, 2) +
                Math.pow(p61.y - p291.y, 2) +
                Math.pow(p61.z - p291.z, 2)
            );

            const r1 = eyeDist / (noseToChin || 1);
            const r2 = eyeDist / (mouthWidth || 1);
            const r3 = noseToChin / (mouthWidth || 1);

            // Fetch stored database
            const db = getFaceDB();
            let identity = 'Unknown';
            let bestConfidence = 0.0;
            let minDistance = 999.0;

            db.forEach((profile) => {
              const [f1, f2, f3] = profile.fingerprint;
              const dist = Math.sqrt(
                Math.pow(r1 - f1, 2) + Math.pow(r2 - f2, 2) + Math.pow(r3 - f3, 2)
              );

              if (dist < minDistance) {
                minDistance = dist;
                if (dist < 0.15) {
                  identity = profile.name;
                  bestConfidence = 1.0 - dist;
                }
              }
            });

            // Calculate bounding box bounds from landmarks
            let minX = 1.0,
              maxX = 0.0,
              minY = 1.0,
              maxY = 0.0;
            landmarks.forEach((lm) => {
              if (lm.x < minX) minX = lm.x;
              if (lm.x > maxX) maxX = lm.x;
              if (lm.y < minY) minY = lm.y;
              if (lm.y > maxY) maxY = lm.y;
            });

            const startX = minX * canvas.width;
            const startY = minY * canvas.height;
            const boxW = (maxX - minX) * canvas.width;
            const boxH = (maxY - minY) * canvas.height;

            faceResult.results.push({
              identity,
              confidence: identity === 'Unknown' ? 0.0 : bestConfidence,
              box: [startX, startY, boxW, boxH],
            });
          }
        }

        setActiveFaceLandmarks(faceLandmarksData);

        // 2. Process Hand Gesture Recognition
        let gestureResult = { landmarks: [] as any[], label: 'None', translated_text: 'None' };
        if (handLandmarker) {
          const handDetections = handLandmarker.detectForVideo(video, timestamp);
          if (handDetections.landmarks && handDetections.landmarks.length > 0) {
            gestureResult.landmarks = handDetections.landmarks;
            const primaryHand = handDetections.landmarks[0];
            const gestureLabel = classifyGestureClient(primaryHand);
            gestureResult.label = gestureLabel;
            gestureResult.translated_text = getTranslatedText(gestureLabel, langRef.current);
          }
        }

        // 3. Fallback Demo Mode (if no model loaded)
        if (!faceLandmarker && !handLandmarker) {
          faceResult = {
            status: 'success',
            results: [
              { identity: 'Demo Mode (Guest)', confidence: 0.95, box: [180, 100, 280, 280] },
            ],
          };

          // Generate mock hand landmarks in client-side demo mode
          const mockLandmarks = [];
          const base_x = 0.5,
            base_y = 0.6;
          for (let i = 0; i < 21; i++) {
            mockLandmarks.push({
              x: base_x + 0.08 * Math.sin(i * 0.5),
              y: base_y - 0.01 * i + 0.03 * Math.cos(i * 0.5),
              z: -0.01 * i,
            });
          }
          gestureResult = {
            landmarks: [mockLandmarks],
            label: 'Hello',
            translated_text: getTranslatedText('Hello', langRef.current),
          };
        }

        setRecognitionResult({
          face: faceResult,
          gesture: gestureResult,
          emotion: { status: 'success', emotion: 'Happy' },
        });
      }

      animationFrameId = requestAnimationFrame(processFrame);
    };

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
        });
        activeStream = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch((err) => console.error('Video feed play fail:', err));
        }
        await initMediaPipe();
        animationFrameId = requestAnimationFrame(processFrame);
      } catch (err) {
        console.error('Camera access failed in client mode:', err);
      }
    };

    startCamera();

    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (activeStream) {
        activeStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [isClientMode]);

  // Non-Client Mode (Socket Server Loop)
  useEffect(() => {
    if (isClientMode) return;

    let intervalId: any = null;
    let activeStream: MediaStream | null = null;

    const startStream = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        activeStream = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch((err) => console.error('Socket video play error:', err));
        }

        intervalId = setInterval(() => {
          const canvas = canvasRef.current;
          if (canvas && videoRef.current) {
            const context = canvas.getContext('2d');
            if (context) {
              context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
              const frame = canvas.toDataURL('image/jpeg');
              if (socketRef.current && socketRef.current.connected) {
                socketRef.current.emit('frame', { frame, lang: langRef.current });
              }
            }
          }
        }, 100);
      } catch (error) {
        console.error('Camera stream fail in socket mode:', error);
      }
    };

    startStream();

    return () => {
      if (intervalId) clearInterval(intervalId);
      if (activeStream) {
        activeStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [isClientMode]);

  // Enroll Face Functionality
  const enrollFace = (name: string): boolean => {
    if (!activeFaceLandmarks) {
      console.warn('Cannot enroll face: No face currently detected in feed.');
      return false;
    }

    try {
      const landmarks = activeFaceLandmarks;
      // Calculate ratios
      const p33 = landmarks[33];
      const p263 = landmarks[263];
      const eyeDist = Math.sqrt(
        Math.pow(p33.x - p263.x, 2) + Math.pow(p33.y - p263.y, 2) + Math.pow(p33.z - p263.z, 2)
      );

      const p4 = landmarks[4];
      const p152 = landmarks[152];
      const noseToChin = Math.sqrt(
        Math.pow(p4.x - p152.x, 2) + Math.pow(p4.y - p152.y, 2) + Math.pow(p4.z - p152.z, 2)
      );

      const p61 = landmarks[61];
      const p291 = landmarks[291];
      const mouthWidth = Math.sqrt(
        Math.pow(p61.x - p291.x, 2) + Math.pow(p61.y - p291.y, 2) + Math.pow(p61.z - p291.z, 2)
      );

      const r1 = eyeDist / (noseToChin || 1);
      const r2 = eyeDist / (mouthWidth || 1);
      const r3 = noseToChin / (mouthWidth || 1);

      const db = getFaceDB();
      // Remove any existing profile with the same name
      const cleanDb = db.filter((p) => p.name.toLowerCase() !== name.toLowerCase());
      cleanDb.push({
        name,
        fingerprint: [r1, r2, r3],
      });

      saveFaceDB(cleanDb);
      console.log(`Enrolled face fingerprint for ${name} successfully.`);
      return true;
    } catch (err) {
      console.error('Error during client-side face enrollment:', err);
      return false;
    }
  };

  return {
    videoRef,
    canvasRef,
    recognitionResult,
    isClientMode,
    modelsLoaded,
    loadingText,
    enrollFace,
  };
};
