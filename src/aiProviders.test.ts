import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  loadAISettings,
  saveAISettings,
  isProviderConfigured,
  testProviderConnection,
  DEFAULT_AI_SETTINGS,
  type AISettings,
} from './aiProviders';

describe('aiProviders settings storage', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('returns defaults (Ollama, no key required) when nothing is saved', () => {
    const settings = loadAISettings();
    expect(settings.provider).toBe('ollama');
    expect(settings.apiKey).toBe('');
    expect(isProviderConfigured(settings)).toBe(true);
  });

  it('round-trips saved settings through localStorage', () => {
    const saved: AISettings = {
      provider: 'openai',
      apiKey: 'sk-test',
      model: 'gpt-4o-mini',
      baseUrl: 'http://localhost:11434',
    };
    saveAISettings(saved);
    expect(loadAISettings()).toEqual(saved);
  });

  it('falls back to defaults if localStorage contains malformed JSON', () => {
    localStorage.setItem('signbridge_ai_settings', '{not valid json');
    expect(loadAISettings()).toEqual(DEFAULT_AI_SETTINGS);
  });
});

describe('isProviderConfigured', () => {
  it('Ollama is considered configured without an API key', () => {
    expect(
      isProviderConfigured({ provider: 'ollama', apiKey: '', model: 'llama3.2', baseUrl: '' })
    ).toBe(true);
  });

  it('cloud providers require a non-empty API key', () => {
    expect(
      isProviderConfigured({ provider: 'openai', apiKey: '', model: 'gpt-4o-mini', baseUrl: '' })
    ).toBe(false);
    expect(
      isProviderConfigured({
        provider: 'openai',
        apiKey: 'sk-x',
        model: 'gpt-4o-mini',
        baseUrl: '',
      })
    ).toBe(true);
  });

  it('requires a non-empty model regardless of provider', () => {
    expect(isProviderConfigured({ provider: 'ollama', apiKey: '', model: '', baseUrl: '' })).toBe(
      false
    );
  });
});

describe('testProviderConnection', () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it('reports success when Ollama is reachable and the model is installed', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ models: [{ name: 'llama3.2:latest' }] }),
    }) as unknown as typeof fetch;

    const result = await testProviderConnection({
      provider: 'ollama',
      apiKey: '',
      model: 'llama3.2',
      baseUrl: 'http://localhost:11434',
    });

    expect(result.ok).toBe(true);
    expect(result.message).toMatch(/available locally/);
  });

  it('reports a network/CORS failure gracefully when Ollama is unreachable', async () => {
    global.fetch = vi
      .fn()
      .mockRejectedValue(new TypeError('Failed to fetch')) as unknown as typeof fetch;

    const result = await testProviderConnection({
      provider: 'ollama',
      apiKey: '',
      model: 'llama3.2',
      baseUrl: 'http://localhost:11434',
    });

    expect(result.ok).toBe(false);
    expect(result.message).toMatch(/Ollama/);
  });

  it('flags an invalid OpenAI API key from a 401 response', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      json: async () => ({}),
    }) as unknown as typeof fetch;

    const result = await testProviderConnection({
      provider: 'openai',
      apiKey: 'sk-bad',
      model: 'gpt-4o-mini',
      baseUrl: '',
    });

    expect(result.ok).toBe(false);
    expect(result.message).toMatch(/Invalid OpenAI API key/);
  });

  it('requires an API key before attempting to test a cloud provider', async () => {
    const result = await testProviderConnection({
      provider: 'groq',
      apiKey: '',
      model: 'llama-3.3-70b-versatile',
      baseUrl: '',
    });
    expect(result.ok).toBe(false);
    expect(result.message).toMatch(/API key/);
  });
});
