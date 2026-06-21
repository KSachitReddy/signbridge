import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), ...(!process.env.VITEST ? [tailwindcss()] : [])],
  test: {
    globals: true,
    environment: 'jsdom',
    coverage: {
      thresholds: {
        statements: 0,
        branches: 0,
        functions: 0,
        lines: 0,
      },
    },
  },
});
