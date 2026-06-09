import { GestureEstimator, Gestures } from 'fingerpose';

// Define a sample custom gesture (e.g., 'Victory')
const victoryGesture = new Gestures.VictoryGesture('victory');

const estimator = new GestureEstimator([
  victoryGesture,
  // Add more custom gestures here
]);

export const estimateCustomGesture = (landmarks: {x: number, y: number, z: number}[]) => {
  // Fingerpose expects landmarks in a specific format
  const formattedLandmarks = landmarks.map(l => [l.x * 640, l.y * 480, l.z * 100]);
  
  const estimation = estimator.estimate(formattedLandmarks, 9);
  
  if (estimation.gestures.length > 0) {
    const bestGesture = estimation.gestures.reduce((p: {confidence: number, name: string}, c: {confidence: number, name: string}) => (p.confidence > c.confidence ? p : c));
    return bestGesture.name;
  }
  return null;
};
