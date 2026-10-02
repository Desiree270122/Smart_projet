import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// En développement (npm run dev), l'interface tourne sur le port 5173 et
// transmet les appels /api au serveur de calcul (uvicorn, port 8000).
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
  build: { chunkSizeWarningLimit: 6000 },
});
