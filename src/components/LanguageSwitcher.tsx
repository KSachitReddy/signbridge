import React from 'react';
import { useTranslation } from 'react-i18next';

export const LanguageSwitcher: React.FC = () => {
  const { i18n } = useTranslation();

  const changeLanguage = (lng: string) => {
    i18n.changeLanguage(lng);
  };

  const languages = [
    { code: 'en', label: 'EN' },
    { code: 'te', label: 'TE' },
    { code: 'hi', label: 'HI' },
    { code: 'ta', label: 'TA' },
    { code: 'kn', label: 'KN' },
    { code: 'ml', label: 'ML' },
    { code: 'tcy', label: 'TCY' },
  ];

  return (
    <div className="lang-switcher">
      {languages.map((lang) => (
        <button 
          key={lang.code} 
          onClick={() => changeLanguage(lang.code)}
          className={i18n.language === lang.code ? 'active' : ''}
        >
          {lang.label}
        </button>
      ))}
    </div>
  );
};
