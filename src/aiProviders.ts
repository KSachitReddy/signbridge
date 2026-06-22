// BYOK (Bring Your Own Key) AI provider integration.
// All requests are made directly from the browser using the user's own API key,
// so they only work for providers that allow direct browser calls (CORS).
// Anthropic requires the `anthropic-dangerous-direct-browser-access` header for this;
// Ollama requires the local server's OLLAMA_ORIGINS to permit this site's origin.

export type ProviderId = 'ollama' | 'openai' | 'gemini' | 'anthropic' | 'groq';

interface ProviderConfig {
  id: ProviderId;
  label: string;
  defaultModel: string;
  suggestedModels: string[];
  requiresApiKey: boolean;
  keyPlaceholder: string;
}

export const PROVIDERS: ProviderConfig[] = [
  {
    id: 'ollama',
    label: 'Ollama (Local)',
    defaultModel: 'llama3.2',
    suggestedModels: ['llama3.2', 'llama3.1', 'mistral', 'gemma2'],
    requiresApiKey: false,
    keyPlaceholder: '',
  },
  {
    id: 'openai',
    label: 'OpenAI',
    defaultModel: 'gpt-4o-mini',
    suggestedModels: ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1-mini', 'o4-mini'],
    requiresApiKey: true,
    keyPlaceholder: 'sk-...',
  },
  {
    id: 'gemini',
    label: 'Google Gemini',
    defaultModel: 'gemini-2.0-flash',
    suggestedModels: ['gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-flash'],
    requiresApiKey: true,
    keyPlaceholder: 'AIza...',
  },
  {
    id: 'anthropic',
    label: 'Anthropic Claude',
    defaultModel: 'claude-3-5-haiku-latest',
    suggestedModels: [
      'claude-3-5-haiku-latest',
      'claude-3-5-sonnet-latest',
      'claude-3-7-sonnet-latest',
    ],
    requiresApiKey: true,
    keyPlaceholder: 'sk-ant-...',
  },
  {
    id: 'groq',
    label: 'Groq',
    defaultModel: 'llama-3.3-70b-versatile',
    suggestedModels: ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768'],
    requiresApiKey: true,
    keyPlaceholder: 'gsk_...',
  },
];

export const getProviderConfig = (id: ProviderId): ProviderConfig =>
  PROVIDERS.find((p) => p.id === id) ?? PROVIDERS[0];

export interface AISettings {
  provider: ProviderId;
  apiKey: string;
  model: string;
  baseUrl: string;
}

const STORAGE_KEY = 'signbridge_ai_settings';
const DEFAULT_OLLAMA_BASE_URL = 'http://localhost:11434';

export const DEFAULT_AI_SETTINGS: AISettings = {
  provider: 'ollama',
  apiKey: '',
  model: getProviderConfig('ollama').defaultModel,
  baseUrl: DEFAULT_OLLAMA_BASE_URL,
};

export const loadAISettings = (): AISettings => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_AI_SETTINGS };
    const parsed = JSON.parse(raw);
    const provider: ProviderId = parsed.provider ?? DEFAULT_AI_SETTINGS.provider;
    return {
      provider,
      apiKey: typeof parsed.apiKey === 'string' ? parsed.apiKey : '',
      model:
        typeof parsed.model === 'string' && parsed.model
          ? parsed.model
          : getProviderConfig(provider).defaultModel,
      baseUrl:
        typeof parsed.baseUrl === 'string' && parsed.baseUrl
          ? parsed.baseUrl
          : DEFAULT_OLLAMA_BASE_URL,
    };
  } catch {
    return { ...DEFAULT_AI_SETTINGS };
  }
};

export const saveAISettings = (settings: AISettings): void => {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
};

// Whether the provider has enough configuration to actually be called:
// Ollama needs no key, cloud providers need a non-empty API key.
export const isProviderConfigured = (settings: AISettings): boolean => {
  if (!settings.model.trim()) return false;
  const config = getProviderConfig(settings.provider);
  if (!config.requiresApiKey) return true;
  return settings.apiKey.trim().length > 0;
};

export interface ConnectionTestResult {
  ok: boolean;
  message: string;
}

const TEST_TIMEOUT_MS = 8000;
// Generous: local models (e.g. Ollama) can take several seconds to cold-load before generating.
const ENHANCE_TIMEOUT_MS = 20000;

const withTimeout = (ms: number) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), ms);
  return { signal: controller.signal, clear: () => clearTimeout(timeoutId) };
};

const describeFetchError = (err: unknown, providerLabel: string, hint?: string): string => {
  if (err instanceof DOMException && err.name === 'AbortError') {
    return `${providerLabel} did not respond in time (timed out).`;
  }
  const base = `Could not reach ${providerLabel}. This is usually a network or CORS restriction`;
  return hint ? `${base}. ${hint}` : `${base}.`;
};

export async function testProviderConnection(settings: AISettings): Promise<ConnectionTestResult> {
  const config = getProviderConfig(settings.provider);

  if (!isProviderConfigured(settings)) {
    return {
      ok: false,
      message: config.requiresApiKey ? 'Enter an API key first.' : 'Enter a model name first.',
    };
  }

  const { signal, clear } = withTimeout(TEST_TIMEOUT_MS);
  try {
    switch (settings.provider) {
      case 'ollama': {
        const base = settings.baseUrl.trim().replace(/\/$/, '') || DEFAULT_OLLAMA_BASE_URL;
        const res = await fetch(`${base}/api/tags`, { signal });
        if (!res.ok) return { ok: false, message: `Ollama responded with status ${res.status}.` };
        const data = await res.json();
        const models: string[] = (data.models ?? []).map((m: { name: string }) => m.name);
        const hasModel = models.some(
          (name) => name === settings.model || name.startsWith(`${settings.model}:`)
        );
        return hasModel
          ? { ok: true, message: `Connected. "${settings.model}" is available locally.` }
          : {
              ok: true,
              message: `Connected, but "${settings.model}" was not found. Run "ollama pull ${settings.model}".`,
            };
      }
      case 'openai': {
        const res = await fetch('https://api.openai.com/v1/models', {
          headers: { Authorization: `Bearer ${settings.apiKey.trim()}` },
          signal,
        });
        if (res.status === 401) return { ok: false, message: 'Invalid OpenAI API key.' };
        if (!res.ok) return { ok: false, message: `OpenAI responded with status ${res.status}.` };
        return { ok: true, message: 'Connected to OpenAI.' };
      }
      case 'gemini': {
        const res = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models?key=${encodeURIComponent(settings.apiKey.trim())}`,
          { signal }
        );
        if (res.status === 400 || res.status === 403)
          return { ok: false, message: 'Invalid Gemini API key.' };
        if (!res.ok) return { ok: false, message: `Gemini responded with status ${res.status}.` };
        return { ok: true, message: 'Connected to Google Gemini.' };
      }
      case 'anthropic': {
        const res = await fetch('https://api.anthropic.com/v1/models', {
          headers: {
            'x-api-key': settings.apiKey.trim(),
            'anthropic-version': '2023-06-01',
            'anthropic-dangerous-direct-browser-access': 'true',
          },
          signal,
        });
        if (res.status === 401) return { ok: false, message: 'Invalid Anthropic API key.' };
        if (!res.ok)
          return { ok: false, message: `Anthropic responded with status ${res.status}.` };
        return { ok: true, message: 'Connected to Anthropic.' };
      }
      case 'groq': {
        const res = await fetch('https://api.groq.com/openai/v1/models', {
          headers: { Authorization: `Bearer ${settings.apiKey.trim()}` },
          signal,
        });
        if (res.status === 401) return { ok: false, message: 'Invalid Groq API key.' };
        if (!res.ok) return { ok: false, message: `Groq responded with status ${res.status}.` };
        return { ok: true, message: 'Connected to Groq.' };
      }
      default:
        return { ok: false, message: 'Unknown provider.' };
    }
  } catch (err) {
    const hint =
      settings.provider === 'ollama'
        ? 'Make sure Ollama is running locally and OLLAMA_ORIGINS allows this site.'
        : undefined;
    return { ok: false, message: describeFetchError(err, config.label, hint) };
  } finally {
    clear();
  }
}

const LANGUAGE_NAMES: Record<string, string> = {
  en: 'English',
  hi: 'Hindi',
  te: 'Telugu',
  ta: 'Tamil',
  kn: 'Kannada',
  ml: 'Malayalam',
  tcy: 'Tulu',
};

const resolveLanguageName = (languageCode: string): string => {
  const lower = languageCode.toLowerCase();
  return LANGUAGE_NAMES[lower] || LANGUAGE_NAMES[lower.slice(0, 2)] || 'English';
};

const buildPrompt = (gestureLabel: string, languageCode: string): string => {
  const languageName = resolveLanguageName(languageCode);
  return (
    `You are an assistive interpreter for Indian Sign Language. A user just signed the gesture ` +
    `"${gestureLabel}". Reply with ONLY one short, natural ${languageName} sentence a hearing ` +
    `person would say to convey this - no explanation, no quotes.`
  );
};

async function chatComplete(
  settings: AISettings,
  prompt: string,
  signal: AbortSignal
): Promise<string | null> {
  switch (settings.provider) {
    case 'ollama': {
      const base = settings.baseUrl.trim().replace(/\/$/, '') || DEFAULT_OLLAMA_BASE_URL;
      const res = await fetch(`${base}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: settings.model,
          messages: [{ role: 'user', content: prompt }],
          stream: false,
        }),
        signal,
      });
      if (!res.ok) return null;
      const data = await res.json();
      return data?.message?.content?.trim() || null;
    }
    case 'openai':
    case 'groq': {
      const url =
        settings.provider === 'openai'
          ? 'https://api.openai.com/v1/chat/completions'
          : 'https://api.groq.com/openai/v1/chat/completions';
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${settings.apiKey.trim()}`,
        },
        body: JSON.stringify({
          model: settings.model,
          messages: [{ role: 'user', content: prompt }],
          max_tokens: 60,
          temperature: 0.4,
        }),
        signal,
      });
      if (!res.ok) return null;
      const data = await res.json();
      return data?.choices?.[0]?.message?.content?.trim() || null;
    }
    case 'gemini': {
      const res = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
          settings.model
        )}:generateContent?key=${encodeURIComponent(settings.apiKey.trim())}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { maxOutputTokens: 60, temperature: 0.4 },
          }),
          signal,
        }
      );
      if (!res.ok) return null;
      const data = await res.json();
      return data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || null;
    }
    case 'anthropic': {
      const res = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': settings.apiKey.trim(),
          'anthropic-version': '2023-06-01',
          'anthropic-dangerous-direct-browser-access': 'true',
        },
        body: JSON.stringify({
          model: settings.model,
          max_tokens: 60,
          messages: [{ role: 'user', content: prompt }],
        }),
        signal,
      });
      if (!res.ok) return null;
      const data = await res.json();
      return data?.content?.[0]?.text?.trim() || null;
    }
    default:
      return null;
  }
}

// Asks the configured BYOK provider for a richer, natural-language interpretation of a
// detected gesture. Returns null (never throws) when unconfigured, unreachable, or erroring -
// callers should fall back to the static dictionary translation in that case.
export async function enhanceTranslation(
  settings: AISettings,
  gestureLabel: string,
  languageCode: string
): Promise<string | null> {
  if (!isProviderConfigured(settings)) return null;

  const { signal, clear } = withTimeout(ENHANCE_TIMEOUT_MS);
  try {
    const prompt = buildPrompt(gestureLabel, languageCode);
    return await chatComplete(settings, prompt, signal);
  } catch {
    return null;
  } finally {
    clear();
  }
}
