import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

const SettingsPage = () => {
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();

  const [lang, setLang] = useState(localStorage.getItem('language') || 'en');
  const [provider, setProvider] = useState(localStorage.getItem('signbridge_provider') || '');
  const [apiKey, setApiKey] = useState(localStorage.getItem('signbridge_apiKey') || '');
  const [endpoint, setEndpoint] = useState(localStorage.getItem('signbridge_endpoint') || '');
  const [model, setModel] = useState(localStorage.getItem('signbridge_model') || '');
  const [notification, setNotification] = useState<string | null>(null);

  useEffect(() => {
    if (provider === 'ollama' && !endpoint) {
      setEndpoint('http://localhost:11434');
      setModel('llama3');
    } else if (provider === 'openai' && !endpoint) {
      setEndpoint('https://api.openai.com/v1');
      setModel('gpt-4o-mini');
    } else if (provider === 'gemini' && !endpoint) {
      setEndpoint('https://generativelanguage.googleapis.com');
      setModel('gemini-1.5-flash');
    } else if (!provider) {
      setEndpoint('');
      setModel('');
    }
  }, [provider]);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();

    localStorage.setItem('language', lang);
    i18n.changeLanguage(lang);

    localStorage.setItem('signbridge_provider', provider);
    localStorage.setItem('signbridge_apiKey', apiKey);
    localStorage.setItem('signbridge_endpoint', endpoint);
    localStorage.setItem('signbridge_model', model);

    setNotification(t('settings.saved'));
    setTimeout(() => setNotification(null), 3000);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6 flex items-center justify-center">
      <div className="w-full max-w-2xl bg-gray-800 border border-gray-700 rounded-2xl shadow-2xl p-8 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/10 rounded-full blur-3xl pointer-events-none"></div>
        <div className="absolute bottom-0 left-0 w-32 h-32 bg-purple-500/10 rounded-full blur-3xl pointer-events-none"></div>

        <div className="flex items-center justify-between mb-8 pb-4 border-b border-gray-700">
          <h1 className="text-3xl font-extrabold text-blue-400 flex items-center gap-2">
            ⚙️ {t('settings.title')}
          </h1>
          <button
            onClick={() => navigate('/camera')}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm transition cursor-pointer"
          >
            ← {t('dashboard.back')}
          </button>
        </div>

        {notification && (
          <div className="mb-6 p-4 bg-green-500/20 border border-green-500 text-green-300 rounded-xl text-center font-medium animate-pulse">
            {notification}
          </div>
        )}

        <form onSubmit={handleSave} className="space-y-6">
          <div className="space-y-2">
            <label className="block text-sm font-semibold text-gray-300">
              {t('settings.language')}
            </label>
            <select
              value={lang}
              onChange={(e) => setLang(e.target.value)}
              className="w-full bg-gray-900 border border-gray-600 rounded-xl p-3 text-white focus:outline-none focus:border-blue-500 transition"
            >
              <option value="en">English</option>
              <option value="hi">हिन्दी (Hindi)</option>
              <option value="te">తెలుగు (Telugu)</option>
            </select>
          </div>

          <hr className="border-gray-700" />

          <div className="space-y-4">
            <div>
              <h2 className="text-xl font-bold text-blue-400">
                {t('settings.byokHeading')}
              </h2>
              <p className="text-sm text-gray-400">
                {t('settings.byokDescription')}
              </p>
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-semibold text-gray-300">
                {t('settings.aiProvider')}
              </label>
              <select
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
                className="w-full bg-gray-900 border border-gray-600 rounded-xl p-3 text-white focus:outline-none focus:border-blue-500 transition"
              >
                <option value="">None (Local translation only)</option>
                <option value="ollama">Ollama (Local LLM)</option>
                <option value="openai">OpenAI (BYOK)</option>
                <option value="gemini">Google Gemini (BYOK)</option>
              </select>
            </div>

            {provider && (
              <div className="space-y-4">
                {provider !== 'ollama' && (
                  <div className="space-y-2">
                    <label className="block text-sm font-semibold text-gray-300">
                      {t('settings.apiKey')}
                    </label>
                    <input
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder={t('settings.apiKeyPlaceholder')}
                      className="w-full bg-gray-900 border border-gray-600 rounded-xl p-3 text-white focus:outline-none focus:border-blue-500 transition"
                      required
                    />
                  </div>
                )}

                <div className="space-y-2">
                  <label className="block text-sm font-semibold text-gray-300">
                    {t('settings.endpoint')}
                  </label>
                  <input
                    type="text"
                    value={endpoint}
                    onChange={(e) => setEndpoint(e.target.value)}
                    placeholder={t('settings.endpointPlaceholder')}
                    className="w-full bg-gray-900 border border-gray-600 rounded-xl p-3 text-white focus:outline-none focus:border-blue-500 transition"
                    required
                  />
                </div>

                <div className="space-y-2">
                  <label className="block text-sm font-semibold text-gray-300">
                    {t('settings.model')}
                  </label>
                  <input
                    type="text"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    placeholder={t('settings.modelPlaceholder')}
                    className="w-full bg-gray-900 border border-gray-600 rounded-xl p-3 text-white focus:outline-none focus:border-blue-500 transition"
                    required
                  />
                </div>
              </div>
            )}
          </div>

          <button
            type="submit"
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-xl shadow-lg hover:shadow-blue-500/20 transform hover:-translate-y-0.5 transition duration-150 cursor-pointer"
          >
            💾 {t('settings.save')}
          </button>
        </form>
      </div>
    </div>
  );
};

export default SettingsPage;
