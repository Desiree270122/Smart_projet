// Comparaison des stratégies EMS : les sept stratégies sur un même cycle (M1 à M6),
// l'explicabilité (E1 à E3), puis la conclusion calculée sur tous les critères.

import { useState } from "react";
import { envoyer, oublier, useDonnees } from "../api.js";
import {
  Attente, Case, Chargement, Depliant, Formule, Graphique, Grille, Legende, Message, PiedNavigation, SelecteurCycle,
  Tableau, Texte, axe, telechargerCsv,
} from "../composants.jsx";
import { useEtat } from "../etat.jsx";
import { useLangue } from "../langue.jsx";

// Les six métriques du protocole.
const METRIQUES = [
  { cle: "energie_km_wh", titre: ["M1 · Énergie consommée", "M1 · Energy consumed"], unite: "Wh/km", echelle: 1, decimales: 1 },
  { cle: "rendement_hess", titre: ["M2 · Rendement du HESS", "M2 · HESS efficiency"], unite: "%", echelle: 100, decimales: 2 },
  { cle: "rmse_delta_soc", titre: ["M3 · Écart entre les SOC", "M3 · Gap between SOCs"], unite: "pts", echelle: 100, decimales: 1 },
  { cle: "pertes_convertisseur_wh", titre: ["M4 · Pertes du convertisseur", "M4 · Converter losses"], unite: "Wh", echelle: 1, decimales: 0 },
  { cle: "violations_totales", titre: ["M5 · Dépassements de limite", "M5 · Limit violations"], unite: "", echelle: 1, decimales: 0 },
  { cle: "rmse_puissance_kw", titre: ["M6 · Écart de puissance", "M6 · Power shortfall"], unite: "kW", echelle: 1, decimales: 2 },
];

function Pastille({ strategie }) {
  return (
    <>
      <span className="pastille" style={{ background: strategie.couleur }} />
      {strategie.nom}
    </>
  );
}

export default function Comparaison() {
  const { tr, nombre, langue } = useLangue();
  const { cycle } = useEtat();
  const [avecExplicabilite, setAvecExplicabilite] = useState(true);
  const [version, setVersion] = useState(0);
  const [mesure, setMesure] = useState({ enCours: false, erreur: null });
  const comparaison = useDonnees(`/cycles/${cycle}/comparaison`, { explicabilite: avecExplicabilite, v: version || undefined });

  const titre = (m) => (m.unite ? `${tr(...m.titre)} (${m.unite})` : tr(...m.titre));
  const colStrategie = tr("Stratégie", "Strategy");

  const mesurerE3 = async () => {
    setMesure({ enCours: true, erreur: null });
    try {
      await envoyer(`/cycles/${cycle}/coherence`, {}, langue);
      oublier(`/cycles/${cycle}/`);
      setVersion((v) => v + 1);
      setMesure({ enCours: false, erreur: null });
    } catch (erreur) {
      setMesure({ enCours: false, erreur: erreur.message });
    }
  };

  const variation = (m, a, b) => {
    if (m.cle === "rendement_hess" || m.cle === "rmse_delta_soc") return `${nombre((b - a) * 100, 2, true)} pts`;
    if (Math.abs(a) < 1e-12) return Math.abs(b) < 1e-12 ? "—" : nombre(b, 2, true);
    return `${nombre(((b - a) / Math.abs(a)) * 100, 1, true)} %`;
  };

  return (
    <>
      <h1>{tr("📊 Comparaison des stratégies EMS", "📊 EMS strategy comparison")}</h1>
      <Legende>
        {tr(
          "Les sept stratégies dans les mêmes conditions : même profil de conduite, mêmes batteries, même convertisseur, même filtre de sécurité. En bas de page : quelle stratégie retenir.",
          "The seven strategies under the same conditions: same driving profile, same batteries, same converter, same safety filter. At the bottom of the page: which strategy to choose."
        )}
      </Legende>
      <div className="rangee">
        <SelecteurCycle />
        <span />
      </div>

      <Depliant titre={tr("Définition des métriques", "Definition of the metrics")}>
        <Texte>
          {tr(
            "- **M1 · Énergie consommée** : énergie tirée des batteries sur le cycle, pertes comprises, rapportée à la distance parcourue.\n- **M2 · Rendement du HESS** : énergie de traction fournie, divisée par cette énergie augmentée des pertes (batteries et convertisseur).\n- **M3 · Écart entre les SOC** : écart quadratique moyen entre les états de charge des deux batteries.\n- **M4 · Pertes du convertisseur** : il ne traite que la puissance (V_EB − V_PB)·I_EB.\n- **M5 · Dépassements de limite** : nombre d'instants où une limite de SOC ou de courant est dépassée.\n- **M6 · Écart de puissance** : écart quadratique moyen entre la puissance fournie et la puissance demandée en traction ; il est nul quand toute la demande est fournie.",
            "- **M1 · Energy consumed**: energy drawn from the batteries over the cycle, losses included, per distance travelled.\n- **M2 · HESS efficiency**: traction energy delivered, divided by that energy plus the losses (batteries and converter).\n- **M3 · Gap between SOCs**: root-mean-square gap between the two batteries' states of charge.\n- **M4 · Converter losses**: it only processes the power (V_EB − V_PB)·I_EB.\n- **M5 · Limit violations**: number of time steps where an SOC or current limit is exceeded.\n- **M6 · Power shortfall**: root-mean-square gap between delivered and demanded traction power; it is zero when the whole demand is met."
          )}
        </Texte>
        <Formule tex={"M_1 = \\frac{1}{D}\\Big(\\int_0^T (P_{EB} + P_{PB})\\,dt + E_{pertes}\\Big) \\qquad M_2 = \\frac{E_{traction}}{E_{traction} + E_{pertes}} \\qquad M_3 = \\sqrt{\\tfrac{1}{N}\\textstyle\\sum_k (SOC_{EB} - SOC_{PB})^2}"} />
        <Formule tex={"M_4 = \\int_0^T (1-\\eta)\\,\\lvert (V_{EB}-V_{PB})\\,I_{EB}\\rvert\\,dt \\qquad M_6 = \\sqrt{\\tfrac{1}{N}\\textstyle\\sum_k (P_{HESS} - P_{dem})^2}"} />
        <Legende>
          {tr(
            "Sauf si les pertes ont été incluses à la simulation (page « Lancer une simulation »), M1, M2 et M4 utilisent des pertes estimées après coup : résistances internes des cellules et rendement mesuré du convertisseur. Ils servent à comparer les stratégies entre elles.",
            "Unless losses were included in the simulation (“Run a simulation” page), M1, M2 and M4 use losses estimated afterwards: internal resistances of the cells and measured converter efficiency. They are meant for comparing the strategies with one another."
          )}
        </Legende>
      </Depliant>

      <Attente etat={comparaison}>
        {(c) => {
          const valeur = (cle, m) => c.metriques[cle][m.cle] * m.echelle;
          const parCle = Object.fromEntries(c.strategies.map((s) => [s.cle, s]));
          return (
            <>
              <h2>{tr("Les six métriques du protocole", "The six metrics of the protocol")}</h2>
              <Tableau
                colonnes={[
                  { cle: "strategie", titre: colStrategie },
                  { cle: "famille", titre: tr("Famille", "Family") },
                  ...METRIQUES.map((m) => ({ cle: m.cle, titre: titre(m), droite: true })),
                ]}
                lignes={c.strategies.map((s) => ({
                  strategie: <Pastille strategie={s} />,
                  famille: s.famille,
                  ...Object.fromEntries(METRIQUES.map((m) => [m.cle, nombre(valeur(s.cle, m), m.decimales)])),
                }))}
              />
              <button
                className="bouton"
                onClick={() =>
                  telechargerCsv(
                    "metriques_M1_M6.csv",
                    [colStrategie, tr("Famille", "Family"), ...METRIQUES.map(titre)],
                    c.strategies.map((s) => [s.nom, s.famille, ...METRIQUES.map((m) => Number(valeur(s.cle, m).toFixed(4)))]),
                    langue
                  )
                }
              >
                {tr("Télécharger le tableau (CSV)", "Download the table (CSV)")}
              </button>

              <Grille colonnes={3}>
                {METRIQUES.map((m) => {
                  const valeurs = c.strategies.map((s) => valeur(s.cle, m));
                  return (
                    <div key={m.cle}>
                      <h4>{titre(m)}</h4>
                      <Graphique
                        hauteur={250}
                        courbes={[{
                          type: "bar", orientation: "h", y: c.strategies.map((s) => s.nom), x: valeurs,
                          marker: { color: c.strategies.map((s) => s.couleur) },
                          text: valeurs.map((v) => nombre(v, m.decimales)), textposition: "outside", cliponaxis: false,
                          hovertemplate: "%{y} : %{text}<extra></extra>",
                        }]}
                        disposition={{
                          showlegend: false, bargap: 0.25, margin: { t: 6, b: 28, l: 8, r: 46 },
                          yaxis: axe({ autorange: "reversed" }), xaxis: axe(),
                        }}
                      />
                    </div>
                  );
                })}
              </Grille>

              <h2>{tr("La demande est-elle fournie, les limites respectées ?", "Is the demand met, are the limits respected?")}</h2>
              <Legende>
                {tr(
                  "Détail de M5 et M6. Une stratégie qui ne fournit pas toute la puissance demandée consomme mécaniquement moins d'énergie : ses valeurs de M1, M2 et M4 sont alors flattées.",
                  "Detail of M5 and M6. A strategy that does not deliver all the demanded power mechanically consumes less energy: its M1, M2 and M4 values are then flattered."
                )}
              </Legende>
              <Tableau
                colonnes={[
                  { cle: "strategie", titre: colStrategie },
                  { cle: "soc", titre: tr("Dépassements de SOC", "SOC violations"), droite: true },
                  { cle: "courant", titre: tr("Dépassements de courant", "Current violations"), droite: true },
                  { cle: "moyen", titre: tr("Écart de puissance moyen (kW)", "Mean power shortfall (kW)"), droite: true },
                  { cle: "max", titre: tr("Écart de puissance maximal (kW)", "Maximum power shortfall (kW)"), droite: true },
                  { cle: "energie", titre: tr("Énergie non fournie (Wh)", "Energy not delivered (Wh)"), droite: true },
                ]}
                lignes={c.strategies.map((s) => {
                  const m = c.metriques[s.cle];
                  return {
                    strategie: <Pastille strategie={s} />, soc: nombre(m.nb_violations, 0), courant: nombre(m.nb_violations_courant, 0),
                    moyen: nombre(m.rmse_puissance_kw, 2), max: nombre(m.ecart_puissance_max_kw, 1), energie: nombre(m.energie_non_servie_wh, 0),
                  };
                })}
              />

              <h2>{tr("Apport de l'approche neuro-symbolique", "Contribution of the neuro-symbolic approach")}</h2>
              <Legende>
                {tr(
                  "Chaque stratégie neuro-symbolique comparée au même réseau de neurones sans connaissances expertes. NS-MLP part des règles expertes, n'apprend qu'une correction limitée et reste sous le contrôle d'un garde-fou de l'ontologie ; NS-LSTM reçoit en plus des états déduits par l'ontologie.",
                  "Each neuro-symbolic strategy compared with the same neural network without expert knowledge. NS-MLP starts from the expert rules, only learns a limited correction and stays under the control of an ontology safeguard; NS-LSTM additionally receives states inferred by the ontology."
                )}
              </Legende>
              <Grille colonnes={Math.max(c.paires.length, 1)}>
                {c.paires.map(([seul, ns]) => (
                  <div key={ns}>
                    <strong>{tr("{a} et {b}", "{a} and {b}", { a: parCle[ns].nom, b: parCle[seul].nom })}</strong>
                    <Tableau
                      colonnes={[
                        { cle: "metrique", titre: tr("Métrique", "Metric") },
                        { cle: "seul", titre: parCle[seul].nom, droite: true },
                        { cle: "ns", titre: parCle[ns].nom, droite: true },
                        { cle: "variation", titre: tr("Variation", "Change"), droite: true },
                      ]}
                      lignes={METRIQUES.map((m) => ({
                        metrique: titre(m), seul: nombre(valeur(seul, m), m.decimales), ns: nombre(valeur(ns, m), m.decimales),
                        variation: variation(m, c.metriques[seul][m.cle], c.metriques[ns][m.cle]),
                      }))}
                    />
                  </div>
                ))}
              </Grille>

              <h2>{tr("Explicabilité", "Explainability")}</h2>
              <Legende>
                {tr(
                  "Peut-on comprendre une décision ? E1 et E2 décrivent ce que permet la construction de la stratégie ; E3 est mesurée : quand on augmente un SOC ou la demande, la décision évolue-t-elle dans le sens attendu par la physique ?",
                  "Can a decision be understood? E1 and E2 describe what the design of the strategy allows; E3 is measured: when an SOC or the demand is increased, does the decision move in the direction expected from physics?"
                )}
              </Legende>
              {!c.e3_disponible ? (
                mesure.enCours ? (
                  <Chargement texte={tr("Mesure de la cohérence physique…", "Measuring physical consistency…")} />
                ) : (
                  <button className="bouton" onClick={mesurerE3}>
                    {tr("Mesurer E3 · cohérence physique (environ 30 s)", "Measure E3 · physical consistency (about 30 s)")}
                  </button>
                )
              ) : null}
              <Message type="erreur">{mesure.erreur}</Message>
              <Tableau
                colonnes={[
                  { cle: "strategie", titre: colStrategie },
                  { cle: "e1", titre: tr("E1 · Comment la décision se lit", "E1 · How the decision can be read") },
                  { cle: "e2", titre: tr("E2 · Comment elle s'explique", "E2 · How it is explained") },
                  { cle: "e3", titre: tr("E3 · Cohérence physique", "E3 · Physical consistency"), droite: true },
                ]}
                lignes={c.strategies.map((s) => {
                  const e = c.explicabilite[s.cle];
                  return { strategie: <Pastille strategie={s} />, e1: e.e1, e2: e.e2, e3: e.e3 === null ? "—" : `${nombre(e.e3 * 100, 0)} %` };
                })}
              />
              {c.e3_disponible ? (
                <Depliant titre={tr("Détail de E3, test par test", "Detail of E3, test by test")}>
                  <Tableau
                    colonnes={[
                      { cle: "strategie", titre: colStrategie },
                      ...c.explicabilite[c.strategies[0].cle].detail.map((d, i) => ({ cle: `t${i}`, titre: d.test, droite: true })),
                    ]}
                    lignes={c.strategies.map((s) => ({
                      strategie: s.nom,
                      ...Object.fromEntries(c.explicabilite[s.cle].detail.map((d, i) => [`t${i}`, `${nombre(d.part * 100, 0)} %`])),
                    }))}
                  />
                  <Legende>
                    {tr(
                      "Sur 120 instants de traction, chaque test augmente une grandeur (SOC de +5 points, demande de +1 kW) et vérifie que la puissance confiée à la batterie Puissance évolue dans le sens indiqué, à 1 % de la demande près.",
                      "Over 120 traction time steps, each test increases one quantity (SOC by +5 points, demand by +1 kW) and checks that the power assigned to the Power battery moves in the stated direction, to within 1 % of the demand."
                    )}
                  </Legende>
                </Depliant>
              ) : null}

              <h2>{tr("Quelle stratégie retenir ?", "Which strategy should be chosen?")}</h2>
              <Legende>
                {tr(
                  "Conclusion calculée sur tous les critères, en trois étapes : (1) une stratégie doit fournir toute la demande et respecter les limites, sinon elle est écartée ; (2) les autres sont comparées deux à deux sur chaque critère ; (3) on vérifie que le résultat ne dépend pas de l'importance donnée à chaque critère.",
                  "Conclusion computed over all criteria, in three steps: (1) a strategy must deliver the whole demand and respect the limits, otherwise it is set aside; (2) the others are compared pairwise on each criterion; (3) we check that the result does not depend on the importance given to each criterion."
                )}
              </Legende>
              <Case
                titre={tr(
                  "Tenir compte de l'explicabilité (E1 à E3) en plus des performances (M1 à M4)",
                  "Take explainability (E1 to E3) into account in addition to performance (M1 to M4)"
                )}
                coche={avecExplicabilite} onChange={setAvecExplicabilite}
              />

              <h3>{tr("Sur ce cycle", "On this cycle")}</h3>
              {avecExplicabilite && !c.e3_disponible ? (
                <Legende>
                  {tr(
                    "E3 n'est pas mesurée pour cette simulation (bouton ci-dessus) : ce critère ne départage pas les stratégies ici.",
                    "E3 has not been measured for this simulation (button above): this criterion does not separate the strategies here."
                  )}
                </Legende>
              ) : null}
              <p>
                {c.sur_ce_cycle.ecartees.length ? (
                  <>
                    {tr("Étape 1 — écartées : ", "Step 1 — set aside: ")}
                    {c.sur_ce_cycle.ecartees.map((e, i) => (
                      <span key={e.strategie.cle}>
                        {i ? tr(" ; ", "; ") : ""}
                        <strong>{e.strategie.nom}</strong> ({e.raison})
                      </span>
                    ))}
                    .
                  </>
                ) : (
                  tr("Étape 1 — toutes les stratégies fournissent la demande et respectent les limites.", "Step 1 — all strategies deliver the demand and respect the limits.")
                )}
              </p>
              {c.sur_ce_cycle.classement.length ? (
                <>
                  <Grille colonnes={2}>
                    <Graphique
                      hauteur={60 + 42 * c.sur_ce_cycle.classement.length}
                      courbes={[{
                        type: "bar", orientation: "h", y: c.sur_ce_cycle.classement.map((l) => l.strategie.nom),
                        x: c.sur_ce_cycle.classement.map((l) => l.score),
                        marker: { color: c.sur_ce_cycle.classement.map((l) => l.strategie.couleur) },
                        text: c.sur_ce_cycle.classement.map((l) => nombre(l.score, 2, true)), textposition: "outside", cliponaxis: false,
                        hovertemplate: "%{y} : %{text}<extra></extra>",
                      }]}
                      disposition={{
                        showlegend: false, margin: { t: 6, b: 44, l: 8, r: 30 },
                        xaxis: axe({ title: { text: tr("Score (comparaisons gagnées − perdues)", "Score (comparisons won − lost)") }, range: [-1.15, 1.15] }),
                        yaxis: axe({ autorange: "reversed" }),
                      }}
                    />
                    <Tableau
                      colonnes={[
                        { cle: "strategie", titre: colStrategie },
                        { cle: "score", titre: tr("Score", "Score"), droite: true },
                        { cle: "tete", titre: tr("En tête dans … des pondérations", "First in … of the weightings"), droite: true },
                        { cle: "forts", titre: tr("Critères où elle n'est battue par aucune autre", "Criteria where no other beats it") },
                      ]}
                      lignes={c.sur_ce_cycle.classement.map((l) => ({
                        strategie: <Pastille strategie={l.strategie} />, score: nombre(l.score, 2, true),
                        tete: `${nombre(l.en_tete * 100, 0)} %`, forts: l.points_forts,
                      }))}
                    />
                  </Grille>
                  <Legende>
                    {tr(
                      "Étapes 2 et 3 — le score va de −1 (battue partout) à +1 (meilleure partout). La colonne suivante chiffre la solidité du résultat : part des 5 000 jeux de poids tirés au hasard où la stratégie arrive première.",
                      "Steps 2 and 3 — the score ranges from −1 (beaten everywhere) to +1 (best everywhere). The next column measures how solid the result is: share of the 5,000 randomly drawn sets of weights in which the strategy comes first."
                    )}
                  </Legende>
                </>
              ) : null}

              {c.conclusion ? (
                <>
                  <h3>{tr("Sur les deux cycles de référence", "On the two reference cycles")}</h3>
                  <Tableau
                    colonnes={[
                      { cle: "strategie", titre: colStrategie },
                      ...c.conclusion.cycles.map((nom, i) => ({ cle: `c${i}`, titre: nom })),
                      { cle: "score", titre: tr("Score moyen", "Mean score"), droite: true },
                      { cle: "tete", titre: tr("En tête dans … des pondérations", "First in … of the weightings"), droite: true },
                    ]}
                    lignes={c.conclusion.lignes.map((l) => ({
                      strategie: <Pastille strategie={l.strategie} />,
                      ...Object.fromEntries(l.cellules.map((cellule, i) => [`c${i}`, cellule])),
                      score: l.score_moyen === null ? "—" : nombre(l.score_moyen, 2, true),
                      tete: l.en_tete === null ? "—" : `${nombre(l.en_tete * 100, 0)} %`,
                    }))}
                  />
                  <Legende>
                    {tr(
                      "Seules les stratégies qui fournissent toute la demande sur les deux cycles sont comparées entre elles ; leurs scores sont recalculés entre elles seules. Le cycle WLTC n'a jamais servi à régler les stratégies à apprentissage.",
                      "Only the strategies that deliver the whole demand on both cycles are compared with one another; their scores are recomputed among themselves only. The WLTC cycle was never used to tune the learning-based strategies."
                    )}
                  </Legende>
                  <Message type={c.conclusion.type}>{c.conclusion.texte}</Message>
                </>
              ) : null}

              <Depliant titre={tr("Méthode et seuils", "Method and thresholds")}>
                <Texte>
                  {tr(
                    "- **Étape 1.** Sont écartées les stratégies qui laissent de la puissance non fournie (M6) ou dépassent une limite de SOC ou de courant (M5).\n- **Étape 2.** Sur chaque critère, une stratégie gagne une comparaison quand elle fait mieux qu'une autre d'un écart supérieur au seuil ci-dessous ; en dessous, les deux sont jugées équivalentes. Score = moyenne, sur les critères, de (comparaisons gagnées − perdues) / nombre d'adversaires. Tous les critères ont le même poids.\n- **Étape 3.** 5 000 jeux de poids sont tirés au hasard ; on compte la part où chaque stratégie est première.\n- M1, M2 et M4 mesurent tous trois des pertes : les performances énergétiques pèsent donc trois critères, l'équilibre des batteries un seul.",
                    "- **Step 1.** Strategies that leave power undelivered (M6) or exceed an SOC or current limit (M5) are set aside.\n- **Step 2.** On each criterion, a strategy wins a comparison when it does better than another by more than the threshold below; under it, the two are judged equivalent. Score = mean, over the criteria, of (comparisons won − lost) / number of opponents. All criteria have the same weight.\n- **Step 3.** 5,000 sets of weights are drawn at random; we count the share in which each strategy comes first.\n- M1, M2 and M4 all measure losses: energy performance therefore counts for three criteria, battery balance for only one."
                  )}
                </Texte>
                <Tableau
                  colonnes={[
                    { cle: "libelle", titre: tr("Critère", "Criterion") },
                    { cle: "sens", titre: tr("Meilleur quand il est", "Better when") },
                    { cle: "seuil", titre: tr("Seuil d'indifférence", "Indifference threshold") },
                  ]}
                  lignes={c.criteres.map((k) => ({
                    libelle: k.libelle, sens: k.sens === "min" ? tr("plus bas", "lower") : tr("plus haut", "higher"), seuil: k.seuil,
                  }))}
                />
                <Legende>
                  {tr(
                    "Méthodes de référence : comparaison par paires à seuils (PROMETHEE II) et analyse de robustesse par tirage des poids (SMAA).",
                    "Reference methods: pairwise comparison with thresholds (PROMETHEE II) and robustness analysis by weight sampling (SMAA)."
                  )}
                </Legende>
              </Depliant>
            </>
          );
        }}
      </Attente>

      <PiedNavigation page="/comparaison" />
    </>
  );
}
