/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50:  "#f0f4ff",
          100: "#dde6ff",
          400: "#6d8fff",
          500: "#4f6ef7",
          600: "#3a57e8",
          700: "#2e45c4",
        },
      },
    },
  },
  plugins: [],
};
