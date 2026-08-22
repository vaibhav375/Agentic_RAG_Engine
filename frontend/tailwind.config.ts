import type { Config } from "tailwindcss";

/**
 * Palette drawn from the thing this page actually shows: an instrument deciding
 * whether it is permitted to answer. Cool measured paper, ink, and three colours
 * that are semantic rather than decorative — grounded, declined, contradicted.
 * Nothing here is an accent for its own sake; if a colour appears, it is a verdict.
 */
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#E9EDEA",       // cool grey-green stock, not cream
        card: "#F7F9F6",
        ink: "#111A19",
        ink2: "#495653",
        ink3: "#5A6764",   // 4.9:1 on paper — AA for the small utility copy
        rule: "#C4CFC9",
        grounded: "#0B5F52",    // supported by the sources
        declined: "#8A5A12",    // refused to answer — a first-class outcome
        contra: "#8E1B2F",      // the sources say otherwise
        live: "#1B4B6B",        // work in progress
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
        sans: ["system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
      },
      letterSpacing: { tightest: "-0.045em" },
    },
  },
  plugins: [],
};
export default config;
