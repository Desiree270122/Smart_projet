// Langue de l'interface (français ou anglais).
//
// Chaque texte est écrit dans les deux langues, là où il est utilisé :
//     const { tr, nombre } = useLangue();
//     tr("La demande vaut {p} kW.", "The demand is {p} kW.", { p: nombre(x, 1) })
// Les phrases calculées (synthèse, raisons, conclusion…) viennent du serveur,
// déjà rédigées dans la langue demandée.

import { createContext, useCallback, useContext, useMemo, useState } from "react";

export const LANGUES = { fr: "Français", en: "English" };
const CLE = "2smart.langue";
const Contexte = createContext(null);

function langueInitiale() {
  const demandee = new URLSearchParams(window.location.search).get("lang");
  if (demandee && LANGUES[demandee.toLowerCase()]) return demandee.toLowerCase();
  const memorisee = localStorage.getItem(CLE);
  if (memorisee && LANGUES[memorisee]) return memorisee;
  return (navigator.language || "fr").toLowerCase().startsWith("fr") ? "fr" : "en";
}

export function FournisseurLangue({ children }) {
  const [langue, setLangue] = useState(langueInitiale);

  const changer = useCallback((code) => {
    localStorage.setItem(CLE, code);
    document.documentElement.lang = code;
    setLangue(code);
  }, []);

  const valeur = useMemo(() => {
    // Texte dans la langue choisie ; les {champs} sont remplis par `valeurs`.
    const tr = (fr, en, valeurs) => {
      const texte = langue === "en" ? en : fr;
      if (!valeurs) return texte;
      return texte.replace(/\{(\w+)\}/g, (_, nom) => (nom in valeurs ? valeurs[nom] : `{${nom}}`));
    };
    // Nombre écrit selon la langue : 12 601,6 en français, 12,601.6 en anglais.
    const nombre = (x, decimales = 1, signe = false) => {
      if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
      return new Intl.NumberFormat(langue === "en" ? "en-US" : "fr-FR", {
        minimumFractionDigits: decimales,
        maximumFractionDigits: decimales,
        signDisplay: signe ? "always" : "auto",
      }).format(Number(x));
    };
    return { langue, changer, tr, nombre, separateurs: langue === "en" ? ".," : ", " };
  }, [langue, changer]);

  return <Contexte.Provider value={valeur}>{children}</Contexte.Provider>;
}

export function useLangue() {
  return useContext(Contexte);
}
