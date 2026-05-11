/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        purple: {
          600: '#9333ea',
          700: '#7e22ce',
        },
      },
    },
  },
  plugins: [],
}
