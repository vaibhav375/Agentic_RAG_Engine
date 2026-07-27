import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0f1116",
        panel: "#171a21",
        edge: "#2a2f3a",
        muted: "#9aa4b2",
        accent: "#7ee0a2",
        accent2: "#8e44ad",
        danger: "#e06c75",
        warn: "#e5c07b",
      },
    },
  },
  plugins: [],
};
export default config;
