// Couleurs des grandeurs physiques : une couleur = une signification, sur toutes
// les pages. Mêmes valeurs que core/style.py (application Streamlit). La couleur
// de chaque stratégie est fournie par le serveur avec la stratégie.

export const COULEURS = {
  demande: "#5B8DEF", // demande de puissance, véhicule
  eb: "#3DBE7A", // batterie Énergie
  pb: "#F0913A", // batterie Puissance
  convertisseur: "#D66BC8",
  decision: "#A98BF5", // décision de l'EMS (alpha)
  violation: "#EF5350",
  reference: "#C5CAD3", // règle de l'ontologie, situation moyenne
  secondaire: "#8B93A7", // repère secondaire (axes, limites)
};

// Rôle d'une étape dans une chaîne de décision -> couleur.
export const COULEURS_ROLES = {
  demande: COULEURS.demande,
  reference: COULEURS.reference,
  secondaire: COULEURS.secondaire,
  decision: COULEURS.decision,
};

export function transparente(hex, opacite) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return `rgba(${r},${g},${b},${opacite})`;
}
