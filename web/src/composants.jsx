// Briques d'affichage communes à toutes les pages.

import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import katex from "katex";
import "katex/dist/katex.min.css";
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import { useDonnees } from "./api.js";
import { COULEURS, COULEURS_ROLES } from "./couleurs.js";
import { useEtat } from "./etat.jsx";
import { useLangue } from "./langue.jsx";
import { PAGES } from "./navigation.js";

const Plot = createPlotlyComponent(Plotly);

// Texte avec mise en forme légère : **gras**, `code`, listes « - … ».
function enLigne(texte) {
  return String(texte)
    .split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
    .filter(Boolean)
    .map((morceau, i) => {
      if (morceau.startsWith("**")) return <strong key={i}>{morceau.slice(2, -2)}</strong>;
      if (morceau.startsWith("`")) return <code key={i}>{morceau.slice(1, -1)}</code>;
      return morceau;
    });
}

export function Texte({ children, className }) {
  if (children === null || children === undefined) return null;
  // Plusieurs morceaux de texte côte à côte arrivent sous forme de liste : on les met bout à bout.
  const lignes = (Array.isArray(children) ? children.filter((c) => c !== null && c !== undefined && c !== false).join("") : String(children)).split("\n");
  if (lignes.length === 1) return <span className={className}>{enLigne(lignes[0])}</span>;
  const blocs = [];
  lignes.forEach((ligne) => {
    const puce = ligne.startsWith("- ");
    const dernier = blocs[blocs.length - 1];
    if (puce && dernier && dernier.liste) dernier.elements.push(ligne.slice(2));
    else if (puce) blocs.push({ liste: true, elements: [ligne.slice(2)] });
    else if (ligne.trim()) blocs.push({ liste: false, texte: ligne });
  });
  return (
    <div className={className}>
      {blocs.map((bloc, i) =>
        bloc.liste ? (
          <ul key={i}>
            {bloc.elements.map((element, j) => (
              <li key={j}>{enLigne(element)}</li>
            ))}
          </ul>
        ) : (
          <p key={i}>{enLigne(bloc.texte)}</p>
        )
      )}
    </div>
  );
}

export function Legende({ children }) {
  return <Texte className="legende">{children}</Texte>;
}

export function Formule({ tex }) {
  const html = useMemo(() => katex.renderToString(tex, { displayMode: true, throwOnError: false }), [tex]);
  return <div className="formule" dangerouslySetInnerHTML={{ __html: html }} />;
}

export function Indicateur({ titre, valeur, aide }) {
  return (
    <div className="indicateur" title={aide || undefined}>
      <div className="indicateur-titre">
        {titre}
        {aide ? <span className="aide"> ⓘ</span> : null}
      </div>
      <div className="indicateur-valeur">{valeur}</div>
    </div>
  );
}

export function Grille({ colonnes = 3, children }) {
  return (
    <div className="grille" style={{ gridTemplateColumns: `repeat(${colonnes}, minmax(0, 1fr))` }}>
      {children}
    </div>
  );
}

// Chaîne d'étapes reliées par des flèches : [{ texte, role }] ou [{ texte, couleur }].
export function Flux({ etapes }) {
  return (
    <div className="flux">
      {etapes.map((etape, i) => {
        const couleur = etape.couleur || COULEURS_ROLES[etape.role] || COULEURS.reference;
        return (
          <span key={i} className="flux-groupe">
            <span className="flux-etape" style={{ borderColor: couleur, color: couleur }}>
              {etape.texte}
            </span>
            {i < etapes.length - 1 ? <span className="flux-fleche">→</span> : null}
          </span>
        );
      })}
    </div>
  );
}

export function Depliant({ titre, children, ouvert = false }) {
  return (
    <details className="depliant" open={ouvert}>
      <summary>{titre}</summary>
      <div className="depliant-contenu">{children}</div>
    </details>
  );
}

export function Carte({ children, className = "" }) {
  return <div className={`carte ${className}`}>{children}</div>;
}

// type : « info », « succes », « alerte » ou « erreur ».
export function Message({ type = "info", children }) {
  if (!children) return null;
  return (
    <div className={`message message-${type}`}>
      <Texte>{children}</Texte>
    </div>
  );
}

export function Chargement({ texte }) {
  const { tr } = useLangue();
  return (
    <div className="chargement">
      <span className="roue" /> {texte || tr("Chargement…", "Loading…")}
    </div>
  );
}

// Affiche l'attente ou l'erreur d'une demande au serveur ; sinon, son contenu.
export function Attente({ etat, texte, children }) {
  if (etat.erreur) return <Message type={etat.erreur.code === 503 ? "alerte" : "erreur"}>{etat.erreur.message}</Message>;
  if (!etat.donnees) return <Chargement texte={texte} />;
  return children(etat.donnees);
}

export function Onglets({ onglets }) {
  const [actif, setActif] = useState(0);
  const courant = onglets[Math.min(actif, onglets.length - 1)];
  return (
    <div className="onglets">
      <div className="onglets-titres" role="tablist">
        {onglets.map((onglet, i) => (
          <button key={onglet.titre} role="tab" aria-selected={i === actif} className={i === actif ? "actif" : ""} onClick={() => setActif(i)}>
            {onglet.titre}
          </button>
        ))}
      </div>
      <div className="onglets-contenu">{courant.contenu()}</div>
    </div>
  );
}

// colonnes : [{ cle, titre, droite }] ; lignes : [{ cle: valeur }]. La première colonne sert d'intitulé de ligne.
export function Tableau({ colonnes, lignes, hauteur }) {
  return (
    <div className="tableau-cadre" style={hauteur ? { maxHeight: hauteur } : undefined}>
      <table className="tableau">
        <thead>
          <tr>
            {colonnes.map((colonne) => (
              <th key={colonne.cle} className={colonne.droite ? "droite" : ""}>
                {colonne.titre}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {lignes.map((ligne, i) => (
            <tr key={i}>
              {colonnes.map((colonne, j) => (
                <td key={colonne.cle} className={`${colonne.droite ? "droite" : ""} ${j === 0 ? "intitule" : ""}`}>
                  {typeof ligne[colonne.cle] === "string" ? <Texte>{ligne[colonne.cle]}</Texte> : ligne[colonne.cle]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Champ({ titre, aide, children }) {
  return (
    <label className="champ" title={aide || undefined}>
      <span className="champ-titre">
        {titre}
        {aide ? <span className="aide"> ⓘ</span> : null}
      </span>
      {children}
    </label>
  );
}

export function Selection({ titre, valeur, options, onChange, aide }) {
  return (
    <Champ titre={titre} aide={aide}>
      <select value={valeur ?? ""} onChange={(e) => onChange(e.target.value)}>
        {options.map((option) => (
          <option key={option.valeur} value={option.valeur}>
            {option.nom}
          </option>
        ))}
      </select>
    </Champ>
  );
}

export function Nombre({ titre, valeur, onChange, min, max, pas, aide, inactif }) {
  return (
    <Champ titre={titre} aide={aide}>
      <input
        type="number" value={valeur} min={min} max={max} step={pas || "any"} disabled={inactif}
        onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))}
      />
    </Champ>
  );
}

export function Curseur({ titre, valeur, onChange, min, max, pas = 1, unite = "", affichage }) {
  return (
    <Champ titre={titre}>
      <div className="curseur">
        <input type="range" value={valeur} min={min} max={max} step={pas} onChange={(e) => onChange(Number(e.target.value))} />
        <span className="curseur-valeur">
          {affichage ? affichage(valeur) : valeur}
          {unite}
        </span>
      </div>
    </Champ>
  );
}

export function Case({ titre, coche, onChange, inactif, aide }) {
  return (
    <label className={`case ${inactif ? "inactif" : ""}`} title={aide || undefined}>
      <input type="checkbox" checked={coche} disabled={inactif} onChange={(e) => onChange && onChange(e.target.checked)} />
      <span>{titre}</span>
    </label>
  );
}

export function Radios({ titre, valeur, options, onChange, aide, inactif }) {
  return (
    <fieldset className="radios" title={aide || undefined} disabled={inactif}>
      {titre ? <legend>{titre}</legend> : null}
      {options.map((option) => (
        <label key={option.valeur} className="case">
          <input type="radio" checked={valeur === option.valeur} onChange={() => onChange(option.valeur)} />
          <span>{option.nom}</span>
        </label>
      ))}
    </fieldset>
  );
}

export function SelecteurCycle() {
  const { tr } = useLangue();
  const { cycles, cycle, choisirCycle } = useEtat();
  return (
    <Selection
      titre={tr("Cycle de conduite", "Driving cycle")} valeur={cycle} onChange={choisirCycle}
      options={cycles.map((c) => ({ valeur: c.id, nom: c.nom }))}
      aide={tr(
        "Résultats affichés sur toutes les pages d'analyse. « WLTC » est un cycle que les stratégies à apprentissage n'ont jamais rencontré.",
        "Results shown on all analysis pages. “WLTC” is a cycle the learning-based strategies have never encountered."
      )}
    />
  );
}

// Carte d'identité du cycle affiché (stratégies simulées, durée…), et stratégie choisie parmi elles.
export function useCycle() {
  const { cycle } = useEtat();
  return useDonnees(cycle ? `/cycles/${cycle}` : null);
}

export function useStrategie(infos, parDefaut = "EMS_power_limitation") {
  const [strategie, setStrategie] = useState(parDefaut);
  const disponibles = infos ? infos.strategies.map((s) => s.cle) : [];
  const valide = disponibles.includes(strategie) ? strategie : disponibles[0] || null;
  useEffect(() => {
    if (valide && valide !== strategie) setStrategie(valide);
  }, [valide, strategie]);
  return [valide, setStrategie];
}

export function SelecteurStrategie({ infos, valeur, onChange }) {
  const { tr } = useLangue();
  return (
    <Selection
      titre={tr("Stratégie", "Strategy")} valeur={valeur} onChange={onChange}
      options={(infos ? infos.strategies : []).map((s) => ({ valeur: s.cle, nom: s.nom }))}
    />
  );
}

// Graphique Plotly au thème sombre de l'application.
const POLICE = { color: "#E6E9EF", family: "'Source Sans 3', 'Segoe UI', system-ui, sans-serif", size: 13 };

export function axe(options = {}) {
  return { gridcolor: "#262C3A", zerolinecolor: "#3A4152", linecolor: "#3A4152", tickcolor: "#3A4152", automargin: true, ...options };
}

export function Graphique({ courbes, disposition, hauteur = 340 }) {
  const { separateurs } = useLangue();
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: POLICE, separators: separateurs,
    margin: { t: 16, b: 44, l: 56, r: 16 }, height: hauteur, autosize: true,
    hoverlabel: { bgcolor: "#161B26", bordercolor: "#3A4152", font: { color: "#F2F4F8", size: 12.5 }, align: "left" },
    legend: { orientation: "h", y: 1.12, x: 0 },
    xaxis: axe(), yaxis: axe(),
    ...disposition,
  };
  return (
    <Plot
      data={courbes} layout={layout} useResizeHandler style={{ width: "100%", height: hauteur }}
      config={{ displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"], displayModeBar: "hover" }}
    />
  );
}

// Courbe invisible qui ajoute à la bulle la phrase d'explication de l'instant survolé.
// lecture : { textes, indices } (les phrases distinctes, et l'indice de celle de chaque instant).
export function courbeExplication(x, y, lecture, { accelere = true, axeY } = {}) {
  return {
    type: accelere ? "scattergl" : "scatter", mode: "lines", x, y, yaxis: axeY, name: "",
    customdata: lecture.indices.map((i) => lecture.textes[i]),
    hovertemplate: "%{customdata}<extra></extra>", line: { color: "rgba(0,0,0,0)", width: 0 }, showlegend: false,
  };
}

export function PiedNavigation({ page }) {
  const { tr } = useLangue();
  const i = PAGES.findIndex((p) => p.chemin === page);
  const precedente = i > 0 ? PAGES[i - 1] : null;
  const suivante = i >= 0 && i < PAGES.length - 1 ? PAGES[i + 1] : null;
  return (
    <nav className="pied">
      {precedente ? (
        <Link className="bouton" to={precedente.chemin}>
          {tr("Précédent : {t}", "Previous: {t}", { t: tr(...precedente.titre) })}
        </Link>
      ) : (
        <span />
      )}
      {suivante ? (
        <Link className="bouton principal" to={suivante.chemin}>
          {tr("Suivant : {t}", "Next: {t}", { t: tr(...suivante.titre) })}
        </Link>
      ) : (
        <span />
      )}
    </nav>
  );
}

// Téléchargement d'un tableau en CSV (séparateur « ; », décimale selon la langue).
export function telechargerCsv(nomFichier, colonnes, lignes, langue) {
  const decimale = langue === "en" ? "." : ",";
  const cellule = (valeur) => {
    if (typeof valeur === "number") return String(valeur).replace(".", decimale);
    const texte = valeur === null || valeur === undefined ? "" : String(valeur);
    return /[;"\n]/.test(texte) ? `"${texte.replace(/"/g, '""')}"` : texte;
  };
  const contenu = [colonnes.join(";"), ...lignes.map((ligne) => ligne.map(cellule).join(";"))].join("\n");
  const lien = document.createElement("a");
  lien.href = URL.createObjectURL(new Blob(["﻿" + contenu], { type: "text/csv;charset=utf-8" }));
  lien.download = nomFichier;
  lien.click();
  URL.revokeObjectURL(lien.href);
}
