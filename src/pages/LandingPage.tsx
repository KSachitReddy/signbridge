import React from 'react';
import { Link } from 'react-router-dom';

const LandingPage: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 text-center">
      <h1 className="text-5xl font-bold text-primary mb-6">Welcome to SignBridge AI</h1>
      <p className="text-xl text-text mb-10 max-w-2xl">
        Bridging communication through real-time sign language recognition, speech-to-sign, and multi-language translation.
      </p>
      <div className="flex gap-4">
        <Link 
          to="/login" 
          className="bg-primary text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 transition"
        >
          Login
        </Link>
        <Link 
          to="/register" 
          className="bg-secondary text-white px-8 py-3 rounded-lg font-semibold hover:bg-green-600 transition"
        >
          Register
        </Link>
        <Link 
          to="/camera" 
          className="bg-gray-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-gray-800 transition"
        >
          Try Demo
        </Link>
      </div>
    </div>
  );
};

export default LandingPage;
