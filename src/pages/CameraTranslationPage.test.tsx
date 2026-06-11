import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import CameraTranslationPage from './CameraTranslationPage';

// Mock the custom hook
vi.mock('../hooks/useSignLanguageAI', () => ({
  useSignLanguageAI: () => ({
    isReady: true,
    recognize: () => ({ gesture: 'Hello', confidence: 0.95 }),
  }),
}));

describe('CameraTranslationPage', () => {
  it('renders the Dashboard header', () => {
    render(<CameraTranslationPage />);
    expect(screen.getByText(/SignBridge AI Dashboard/i)).toBeDefined();
  });
});
