import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

const LandingPage: React.FC = () => {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 text-center bg-gray-900 text-white relative">
      <Link
        to="/settings"
        className="absolute top-6 right-6 p-3 bg-gray-800 hover:bg-gray-700 text-white rounded-full shadow-lg border border-gray-750 transition"
        title="Settings"
      >
        ⚙️
      </Link>

      <h1 className="text-5xl font-extrabold text-blue-400 mb-6">{t('landing.title')}</h1>
      <p className="text-xl text-gray-300 mb-10 max-w-2xl">
        {t('landing.description')}
      </p>
      <div className="flex flex-wrap justify-center gap-4">
        <Link
          to="/login"
          className="bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 transition"
        >
          {t('landing.login')}
        </Link>
        <Link
          to="/register"
          className="bg-green-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-green-700 transition"
        >
          {t('landing.register')}
        </Link>
        <Link
          to="/camera"
          className="bg-gray-750 text-white px-8 py-3 rounded-lg font-semibold hover:bg-gray-800 border border-gray-700 transition"
        >
          {t('landing.tryDemo')}
        </Link>
      </div>
    </div>
  );
};

export default LandingPage;

// Formatted with Prettier
