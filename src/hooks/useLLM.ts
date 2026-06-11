import { useState } from 'react';

export const useLLM = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const translateText = async (text: string, targetLanguage: string): Promise<string> => {
    if (!text) return '';

    const provider = localStorage.getItem('signbridge_provider') || '';
    const apiKey = localStorage.getItem('signbridge_apiKey') || '';
    const endpoint = localStorage.getItem('signbridge_endpoint') || '';
    const model = localStorage.getItem('signbridge_model') || '';

    if (!provider) {
      return text;
    }

    setLoading(true);
    setError(null);

    const systemPrompt = `You are a translation assistant for SignBridge AI, a sign language recognition system.
Translate the following recognized hand gestures or words into a natural, grammatically correct sentence in the target language: ${targetLanguage}.
Do not add any explanation, thoughts, or extra formatting. Output ONLY the translated sentence.`;

    try {
      if (provider === 'ollama') {
        const ollamaUrl = (endpoint || 'http://localhost:11434').replace(/\/$/, '') + '/api/chat';
        const response = await fetch(ollamaUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: model || 'llama3',
            messages: [
              { role: 'system', content: systemPrompt },
              { role: 'user', content: `Refine and translate: "${text}"` }
            ],
            stream: false
          })
        });

        if (!response.ok) throw new Error(`Ollama returned status ${response.status}`);
        const data = await response.json();
        return data.message?.content?.trim() || text;

      } else if (provider === 'openai') {
        const openaiUrl = (endpoint || 'https://api.openai.com/v1').replace(/\/$/, '') + '/chat/completions';
        const response = await fetch(openaiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
          },
          body: JSON.stringify({
            model: model || 'gpt-4o-mini',
            messages: [
              { role: 'system', content: systemPrompt },
              { role: 'user', content: `Refine and translate: "${text}"` }
            ]
          })
        });

        if (!response.ok) throw new Error(`OpenAI returned status ${response.status}`);
        const data = await response.json();
        return data.choices?.[0]?.message?.content?.trim() || text;

      } else if (provider === 'gemini') {
        const geminiModel = model || 'gemini-1.5-flash';
        const baseUrl = (endpoint || 'https://generativelanguage.googleapis.com').replace(/\/$/, '');
        const geminiUrl = `${baseUrl}/v1beta/models/${geminiModel}:generateContent?key=${apiKey}`;
        
        const response = await fetch(geminiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contents: [{
              parts: [{
                text: `${systemPrompt}\n\nRefine and translate: "${text}"`
              }]
            }]
          })
        });

        if (!response.ok) throw new Error(`Gemini returned status ${response.status}`);
        const data = await response.json();
        return data.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || text;
      }

      return text;
    } catch (err: any) {
      console.error('LLM translation error:', err);
      setError(err.message || 'Unknown translation error');
      return text;
    } finally {
      setLoading(false);
    }
  };

  return { translateText, loading, error };
};
