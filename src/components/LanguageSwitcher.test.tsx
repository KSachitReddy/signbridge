import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LanguageSwitcher } from './LanguageSwitcher';
import React from 'react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    i18n: {
      language: 'en',
      changeLanguage: vi.fn(),
    },
  }),
}));

describe('LanguageSwitcher', () => {
  it('renders all language buttons', () => {
    render(<LanguageSwitcher />);
    expect(screen.getByText('EN')).toBeDefined();
    expect(screen.getByText('TE')).toBeDefined();
    expect(screen.getByText('HI')).toBeDefined();
  });
});
