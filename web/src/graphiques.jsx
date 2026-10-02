// Graphiques d'une stratégie sur un cycle (tableau de bord et page Résultats).
// `courbes` est la réponse de /api/cycles/{cycle}/strategies/{stratégie}/courbes.
// Au survol, la bulle donne les valeurs et une phrase en clair sur l'instant visé.

import { COULEURS, transparente } from "./couleurs.js";
import { Graphique, axe, courbeExplication } from "./composants.jsx";
import { useLangue } from "./langue.jsx";

// Portion du cycle affichée à l'ouverture (15 minutes) ; la réglette permet de parcourir le reste.
function vueInitiale(tempsMin) {
  const fin = tempsMin[tempsMin.length - 1];
  const debut = Math.min(60, Math.max(0, fin - 15));
  return [debut, Math.min(debut + 15, fin)];
}

function useTextes() {
  const { tr } = useLangue();
  return {
    tr,
    eb: tr("Batterie Énergie", "Energy battery"),
    pb: tr("Batterie Puissance", "Power battery"),
    conv: tr("Convertisseur", "Converter"),
    temps: tr("Temps (min)", "Time (min)"),
    // Ligne de la bulle d'une courbe : son nom, puis sa valeur.
    bulle: (nom, valeur) => `${nom}${tr(" : ", ": ")}${valeur}<extra></extra>`,
  };
}

function ligneHorizontale(y, texte, { axeY = "y", couleur = COULEURS.secondaire, haut = true } = {}) {
  return {
    forme: { type: "line", xref: "paper", x0: 0, x1: 1, yref: axeY, y0: y, y1: y, line: { color: couleur, dash: "dot", width: 1 } },
    annotation: {
      xref: "paper", x: 0, yref: axeY, y, text: texte, showarrow: false, xanchor: "left",
      yanchor: haut ? "bottom" : "top", font: { size: 11, color: "#C5CAD3" },
    },
  };
}

export function GraphiquePuissances({ courbes, limites }) {
  const { tr, eb, pb, temps, bulle } = useTextes();
  const demande = tr("Demande", "Demand");
  const limite = ligneHorizontale(limites.p_eb_max_w / 1000, tr("limite de la batterie Énergie", "Energy battery limit"));
  return (
    <Graphique
      hauteur={390}
      courbes={[
        // Courbe non accélérée : c'est elle que la réglette reproduit en miniature.
        { type: "scatter", mode: "lines", x: courbes.temps_min, y: courbes.demande_kw, name: demande,
          line: { color: COULEURS.demande, width: 1.2 }, hovertemplate: bulle(demande, "%{y:.1f} kW") },
        { type: "scattergl", mode: "lines", x: courbes.temps_min, y: courbes.eb_kw, name: eb,
          line: { color: COULEURS.eb, width: 1.2 }, hovertemplate: bulle(eb, "%{y:.1f} kW") },
        { type: "scattergl", mode: "lines", x: courbes.temps_min, y: courbes.pb_kw, name: pb,
          line: { color: COULEURS.pb, width: 1.2 }, hovertemplate: bulle(pb, "%{y:.1f} kW") },
        courbeExplication(courbes.temps_min, courbes.demande_kw, courbes.lectures.repartition),
      ]}
      disposition={{
        hovermode: "x unified", margin: { t: 16, b: 52, l: 56, r: 16 },
        yaxis: axe({ title: { text: tr("Puissance (kW)", "Power (kW)") } }),
        xaxis: axe({ title: { text: temps }, rangeslider: { visible: true, thickness: 0.08 }, range: vueInitiale(courbes.temps_min), hoverformat: ".1f" }),
        shapes: [limite.forme], annotations: [limite.annotation],
      }}
    />
  );
}

export function GraphiqueSoc({ courbes, limites, avecMaximum = false }) {
  const { tr, eb, pb, temps, bulle } = useTextes();
  const minimum = ligneHorizontale(limites.soc_eb_min * 100, tr("SOC minimal", "Minimum SOC"), { couleur: COULEURS.violation, haut: false });
  const maximum = ligneHorizontale(limites.soc_max * 100, tr("SOC maximal", "Maximum SOC"), { couleur: COULEURS.reference });
  const lignes = avecMaximum ? [minimum, maximum] : [minimum];
  return (
    <Graphique
      hauteur={320}
      courbes={[
        { type: "scatter", mode: "lines", x: courbes.soc_temps_min, y: courbes.soc_eb_pct, name: eb,
          line: { color: COULEURS.eb, width: 2 }, hovertemplate: bulle(eb, "%{y:.1f} %") },
        { type: "scatter", mode: "lines", x: courbes.soc_temps_min, y: courbes.soc_pb_pct, name: pb,
          line: { color: COULEURS.pb, width: 2 }, hovertemplate: bulle(pb, "%{y:.1f} %") },
        courbeExplication(courbes.soc_temps_min, courbes.soc_eb_pct, courbes.lectures.soc, { accelere: false }),
      ]}
      disposition={{
        hovermode: "x unified",
        xaxis: axe({ title: { text: temps }, hoverformat: ".1f" }),
        yaxis: axe({ title: { text: "SOC (%)" }, range: [0, 105] }),
        shapes: lignes.map((l) => l.forme), annotations: lignes.map((l) => l.annotation),
      }}
    />
  );
}

// Trois graphiques superposés, un par composant : au-dessus de zéro la décharge, en dessous la recharge.
export function GraphiqueChargeDecharge({ courbes, limites }) {
  const { tr, eb, pb, conv, temps, bulle } = useTextes();
  const { nombre } = useLangue();
  const composants = [
    { nom: eb, y: courbes.eb_kw, couleur: COULEURS.eb, lecture: courbes.lectures.eb, axeY: "y",
      max: limites.p_eb_max_w / 1000, min: limites.p_eb_min_w / 1000 },
    { nom: pb, y: courbes.pb_kw, couleur: COULEURS.pb, lecture: courbes.lectures.pb, axeY: "y2",
      max: limites.p_pb_max_w / 1000, min: limites.p_pb_min_w / 1000 },
    { nom: conv, y: courbes.conv_kw, couleur: COULEURS.convertisseur, lecture: courbes.lectures.conv, axeY: "y3",
      max: limites.p_conv_max_w / 1000, min: limites.p_conv_min_w / 1000 },
  ];
  const traces = [];
  const formes = [];
  const annotations = [];
  composants.forEach((c) => {
    const etats = c.lecture.indices.map((i) => c.lecture.textes[i]);
    traces.push({
      type: "scatter", mode: "lines", x: courbes.temps_min, y: c.y.map((v) => Math.max(v, 0)), yaxis: c.axeY, name: c.nom,
      line: { color: c.couleur, width: 1 }, fill: "tozeroy", fillcolor: transparente(c.couleur, 0.55), showlegend: false,
      // La bulle dit ce que fait le composant : décharge, recharge ou repos, et la part de sa limite.
      customdata: etats, hovertemplate: bulle(c.nom, "%{customdata}"),
    });
    traces.push({
      type: "scatter", mode: "lines", x: courbes.temps_min, y: c.y.map((v) => Math.min(v, 0)), yaxis: c.axeY, name: c.nom,
      line: { color: c.couleur, width: 1, dash: "dot" }, fill: "tozeroy", fillcolor: transparente(c.couleur, 0.22),
      showlegend: false, hoverinfo: "skip",
    });
    annotations.push({
      text: `${c.nom} (kW)`, xref: "paper", x: 0.5, yref: `${c.axeY} domain`, y: 1, yanchor: "bottom", showarrow: false,
      font: { size: 13 },
    });
    // Les limites ne sont tracées que si elles sont à l'échelle des puissances atteintes.
    const amplitude = Math.max(...c.y.map((v) => Math.abs(v)), 0.001);
    [
      [c.max, tr("limite de décharge", "discharge limit"), true],
      [c.min, tr("limite de recharge", "charge limit"), false],
    ].forEach(([valeur, texte, haut]) => {
      if (Math.abs(valeur) > 1.5 * amplitude) return;
      const decimales = Math.abs(valeur) < 5 ? 2 : 1;
      const ligne = ligneHorizontale(valeur, `${texte} ${nombre(valeur, decimales)} kW`, { axeY: c.axeY, haut });
      ligne.annotation.font.size = 10;
      formes.push(ligne.forme);
      annotations.push(ligne.annotation);
    });
  });
  return (
    <Graphique
      hauteur={720}
      courbes={traces}
      disposition={{
        hovermode: "x unified", showlegend: false, margin: { t: 30, b: 52, l: 56, r: 16 },
        grid: { rows: 3, columns: 1, subplots: [["xy"], ["xy2"], ["xy3"]], roworder: "top to bottom", ygap: 0.14 },
        xaxis: axe({ title: { text: temps }, rangeslider: { visible: true, thickness: 0.06 }, range: vueInitiale(courbes.temps_min), hoverformat: ".1f" }),
        yaxis: axe(), yaxis2: axe(), yaxis3: axe(),
        shapes: formes, annotations,
      }}
    />
  );
}

export function GraphiquePertes({ courbes }) {
  const { tr, eb, pb, conv, temps, bulle } = useTextes();
  const p = courbes.pertes;
  const valeur = `%{y:.0f} Wh · ${tr("en ce moment", "right now")} %{customdata:.0f} W`;
  const total = tr("Pertes totales", "Total losses");
  const courbe = (nom, y, instantane, couleur, style = {}) => ({
    type: "scatter", mode: "lines", x: p.temps_min, y, name: nom, customdata: instantane,
    line: { color: couleur, width: 1.8, ...style }, hovertemplate: bulle(nom, valeur),
  });
  return (
    <Graphique
      hauteur={320}
      courbes={[
        courbe(eb, p.eb_wh, p.eb_w, COULEURS.eb),
        courbe(pb, p.pb_wh, p.pb_w, COULEURS.pb),
        courbe(conv, p.conv_wh, p.conv_w, COULEURS.convertisseur),
        courbe(total, p.total_wh, p.total_w, COULEURS.reference, { width: 2.2, dash: "dash" }),
      ]}
      disposition={{
        hovermode: "x unified",
        xaxis: axe({ title: { text: temps }, hoverformat: ".1f" }),
        yaxis: axe({ title: { text: tr("Pertes cumulées (Wh)", "Cumulative losses (Wh)") } }),
      }}
    />
  );
}
