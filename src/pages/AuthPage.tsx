import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

const AuthPage = ({ type }: { type: 'login' | 'register' }) => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log(`${type} submitted for:`, email);
    navigate('/camera');
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-4">
      <form onSubmit={handleSubmit} className="bg-gray-800 border border-gray-700 p-8 rounded-2xl shadow-xl w-full max-w-sm">
        <h2 className="text-3xl font-extrabold text-blue-400 mb-6 text-center">
          {type === 'login' ? t('auth.login') : t('auth.register')}
        </h2>
        <input
          type="email"
          placeholder={t('auth.email')}
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full p-3 mb-4 bg-gray-900 border border-gray-650 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 transition"
          required
        />
        <input
          type="password"
          placeholder={t('auth.password')}
          className="w-full p-3 mb-6 bg-gray-900 border border-gray-650 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 transition"
          required
        />
        <button
          type="submit"
          className="w-full bg-blue-650 text-white p-3 rounded-lg font-bold hover:bg-blue-700 transition cursor-pointer"
        >
          {type === 'login' ? t('auth.login') : t('auth.register')}
        </button>
      </form>
    </div>
  );
};

export default AuthPage;

// Formatted with Prettier
