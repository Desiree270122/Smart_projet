// Cadre de l'application : menu à gauche (pages, langue), page à droite.

import { useEffect } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import { Message } from "./composants.jsx";
import { useEtat } from "./etat.jsx";
import { LANGUES, useLangue } from "./langue.jsx";
import { SECTIONS } from "./navigation.js";
import Comparaison from "./pages/Comparaison.jsx";
import Connaissances from "./pages/Connaissances.jsx";
import Explication from "./pages/Explication.jsx";
import Fonctionnement from "./pages/Fonctionnement.jsx";
import Lancer from "./pages/Lancer.jsx";
import Preparer from "./pages/Preparer.jsx";
import Resultats from "./pages/Resultats.jsx";
import TableauDeBord from "./pages/TableauDeBord.jsx";

function Menu() {
  const { langue, changer, tr } = useLangue();
  return (
    <aside className="menu">
      <nav>
        {SECTIONS.map((section, i) => (
          <div key={i} className="menu-section">
            {section.titre ? <div className="menu-titre">{tr(...section.titre)}</div> : null}
            {section.pages.map((page) => (
              <NavLink key={page.chemin} to={page.chemin} end className={({ isActive }) => (isActive ? "actif" : "")}>
                {tr(...page.titre)}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>
      <div className="menu-langue">
        <div className="menu-titre">🌐 Langue / Language</div>
        {Object.entries(LANGUES).map(([code, nom]) => (
          <label key={code} className="case">
            <input type="radio" name="langue" checked={langue === code} onChange={() => changer(code)} />
            <span>{nom}</span>
          </label>
        ))}
      </div>
      <div className="menu-marque">
        <div className="marque">2SMART</div>
        <div>{tr("Gestion intelligente de l'énergie", "Smart energy management")}</div>
        <div className="version">Version 2.0</div>
      </div>
    </aside>
  );
}

export default function App() {
  const { tr } = useLangue();
  const { erreurServeur } = useEtat();
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  return (
    <div className="application">
      <Menu />
      <main className="page">
        {erreurServeur ? (
          <Message type="erreur">
            {tr(
              "Le serveur de calcul ne répond pas. Lancez-le avec : uvicorn api.main:app --port 8000",
              "The calculation server is not responding. Start it with: uvicorn api.main:app --port 8000"
            )}
          </Message>
        ) : (
          <Routes>
            <Route path="/" element={<TableauDeBord />} />
            <Route path="/preparer" element={<Preparer />} />
            <Route path="/lancer" element={<Lancer />} />
            <Route path="/resultats" element={<Resultats />} />
            <Route path="/comparaison" element={<Comparaison />} />
            <Route path="/explication" element={<Explication />} />
            <Route path="/fonctionnement" element={<Fonctionnement />} />
            <Route path="/connaissances" element={<Connaissances />} />
            <Route path="*" element={<TableauDeBord />} />
          </Routes>
        )}
      </main>
    </div>
  );
}
