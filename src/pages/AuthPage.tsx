import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const AuthPage = ({ type }: { type: 'login' | 'register' }) => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log(`${type} submitted for:`, email);
    // Placeholder: Redirect to dashboard after 'mock' auth
    navigate('/camera');
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-background p-4">
      <form onSubmit={handleSubmit} className="bg-white p-8 rounded-xl shadow-md w-full max-w-sm">
        <h2 className="text-2xl font-bold text-primary mb-6 capitalize">{type}</h2>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full p-3 mb-4 border border-gray-300 rounded-lg"
          required
        />
        <input
          type="password"
          placeholder="Password"
          className="w-full p-3 mb-6 border border-gray-300 rounded-lg"
          required
        />
        <button
          type="submit"
          className="w-full bg-primary text-white p-3 rounded-lg font-semibold hover:bg-blue-700"
        >
          {type === 'login' ? 'Sign In' : 'Create Account'}
        </button>
      </form>
    </div>
  );
};

export default AuthPage;

// Formatted with Prettier
