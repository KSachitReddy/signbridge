import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import CameraTranslationPage from './pages/CameraTranslationPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/camera" element={<CameraTranslationPage />} />
        {/* Placeholder for future routes */}
        <Route path="/login" element={<div>Login Page (Coming Soon)</div>} />
        <Route path="/register" element={<div>Register Page (Coming Soon)</div>} />
      </Routes>
    </Router>
  );
}

export default App;
