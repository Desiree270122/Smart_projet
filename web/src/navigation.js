// Menu de l'application : les pages dans l'ordre du parcours, groupées par
// section. Les titres décrivent ce que l'on veut faire (français, anglais).

export const SECTIONS = [
  {
    titre: null,
    pages: [{ chemin: "/", titre: ["🏠 Tableau de bord", "🏠 Dashboard"] }],
  },
  {
    titre: ["📈 Simulation", "📈 Simulation"],
    pages: [
      { chemin: "/preparer", titre: ["📂 Préparer une simulation", "📂 Prepare a simulation"] },
      { chemin: "/lancer", titre: ["▶️ Lancer une simulation", "▶️ Run a simulation"] },
      { chemin: "/resultats", titre: ["📈 Résultats de simulation", "📈 Simulation results"] },
    ],
  },
  {
    titre: ["⚖️ Comparaison", "⚖️ Comparison"],
    pages: [{ chemin: "/comparaison", titre: ["📊 Comparaison des stratégies EMS", "📊 EMS strategy comparison"] }],
  },
  {
    titre: ["🔍 Explication", "🔍 Explanation"],
    pages: [
      { chemin: "/explication", titre: ["💡 Pourquoi cette décision ?", "💡 Why this decision?"] },
      { chemin: "/fonctionnement", titre: ["🧠 Fonctionnement des stratégies EMS", "🧠 How the EMS strategies work"] },
      { chemin: "/connaissances", titre: ["📚 Base de connaissances", "📚 Knowledge base"] },
    ],
  },
];

export const PAGES = SECTIONS.flatMap((section) => section.pages);
