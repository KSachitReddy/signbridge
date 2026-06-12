import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig(async () => {
  const plugins: any[] = [react()];

  if (!process.env.VITEST) {
    try {
      const tailwindcss = (await import('@tailwindcss/vite')).default;
      plugins.push(tailwindcss());
    } catch (e) {
      console.warn('Tailwind CSS plugin could not be loaded:', e);
    }
  }

  return {
    plugins,
    test: {
      globals: true,
      environment: 'jsdom',
    },
  } as any;
});
