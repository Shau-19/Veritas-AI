import type { Config } from "tailwindcss";
const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["'Comic Neue'", "'Comic Sans MS'", "cursive", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
        body: ["'Comic Neue'", "'Comic Sans MS'", "cursive", "sans-serif"],
      },
      colors: {
        bg: "#0f0f12",
        surface: "#181922",
        border: "#000000",
        muted: "#262837",
        dim: "#666b85",
        text: "#ffffff",
        subtle: "#a0a5c0",
        accent: "#ff5c36",
        "accent-dim": "#ba3d1e",
        glow: "#ff5c36",
        green: "#22c55e",
        amber: "#f59e0b",
        red: "#ef4444",
      },
      animation: {
        "fade-up": "fadeUp 0.6s ease forwards",
        "fade-in": "fadeIn 0.4s ease forwards",
        pulse2: "pulse2 3s ease-in-out infinite",
        shimmer: "shimmer 2s linear infinite",
        float: "float 6s ease-in-out infinite",
      },
      keyframes: {
        fadeUp: { "0%": { opacity: "0", transform: "translateY(20px)" }, "100%": { opacity: "1", transform: "translateY(0)" } },
        fadeIn: { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
        pulse2: { "0%,100%": { opacity: "0.4" }, "50%": { opacity: "1" } },
        shimmer: { "0%": { backgroundPosition: "-200% 0" }, "100%": { backgroundPosition: "200% 0" } },
        float: { "0%,100%": { transform: "translateY(0px)" }, "50%": { transform: "translateY(-10px)" } },
      },
    },
  },
  plugins: [],
};
export default config;
