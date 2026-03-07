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
        background: "var(--background)",
        foreground: "var(--foreground)",
        groww: {
          dark: "#0B0F1A",
          card: "#181E29",
          teal: "#00D09C",
          hover: "#00B085",
          border: "#2A3441",
          text: "#E0E5ED",
          muted: "#8A9AAB",
        }
      },
    },
  },
  plugins: [],
};
export default config;
