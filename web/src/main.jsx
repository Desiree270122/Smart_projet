import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";
import { FournisseurEtat } from "./etat.jsx";
import { FournisseurLangue } from "./langue.jsx";
import "./styles.css";

createRoot(document.getElementById("racine")).render(
  <React.StrictMode>
    <BrowserRouter>
      <FournisseurLangue>
        <FournisseurEtat>
          <App />
        </FournisseurEtat>
      </FournisseurLangue>
    </BrowserRouter>
  </React.StrictMode>
);
