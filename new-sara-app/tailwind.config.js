/** @type {import('tailwindcss').Config} */
export default {
    content: [
      "./index.html",
      "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
      extend: {
        colors: {
          'bg-color': '#0f1117',
          'sidebar-bg': '#1a1d28',
          'text-color': '#e2e8f0',
          'muted-color': '#8e8ea0',
          'border-color': '#2d3748',
          'accent-color': '#10a37f',
          'accent-hover': '#0d8a6c',
          'card-bg': '#1e222f',
          'user-bubble': '#2a2e3b',
          'assistant-bubble': '#444654',
          'error-color': '#ef4444',
          'input-bg': '#262b38',
          'code-bg': '#1e1e1e',
          'hover-color': '#2b2d39',
        },
      },
    },
    plugins: [],
  }