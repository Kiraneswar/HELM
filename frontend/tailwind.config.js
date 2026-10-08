/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      colors: {
        space: {
          950: '#03001C',
          900: '#06003C',
          800: '#0D0050',
          700: '#150066',
        },
        violet: {
          400: '#A78BFA',
          500: '#8B5CF6',
          600: '#7C3AED',
        },
        teal: {
          400: '#2DD4BF',
          500: '#14B8A6',
        },
        pink: {
          400: '#F472B6',
          500: '#EC4899',
        },
      },
    },
  },
  plugins: [],
}
