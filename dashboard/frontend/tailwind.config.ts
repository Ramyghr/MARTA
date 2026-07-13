import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: "#0a0a0a",
          surface: "#111111",
          border: "#222222",
          muted: "#333333",
          text: "#e0e0e0",
          dim: "#666666",
          green: "#00ff7f",
          red: "#ff3737",
          yellow: "#ffd700",
          blue: "#4a9eff",
          orange: "#ff8c00",
          cyan: "#00d4ff",
        },
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "'Fira Code'", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
