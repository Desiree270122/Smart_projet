// Pourquoi cette décision ? Pour un instant du cycle et une stratégie : la décision
// prise, trois à cinq raisons en clair, puis le détail par onglets (décision
// décomposée, raisons, connaissances expertes, réseau de neurones, « et si… ? »,
// bilan sur tout le cycle).

import { useEffect, useState } from "react";
import { useDonnees } from "../api.js";
import {
  Attente, Carte, Curseur, Depliant, Flux, Graphique, Grille, Indicateur, Legende, Message, Onglets, PiedNavigation,
  SelecteurCycle, SelecteurStrategie, Tableau, Texte, axe, useCycle, useStrategie,
} from "../composants.jsx";
import { COULEURS } from "../couleurs.js";
import { useEtat } from "../etat.jsx";
import { useLangue } from "../langue.jsx";

function useFormats() {
  const { tr, nombre } = useLangue();
  return {
    tr,
    nombre,
    kw: (w) => `${nombre(w / 1000, 1)} kW`,
    pct: (x, decimales = 1) => `${nombre(x * 100, decimales)} %`,
    partPb: tr("Part confiée à la batterie Puissance (%)", "Share assigned to the Power battery (%)"),
    couleurs: tr(
      "orange = pousse vers la batterie Puissance, vert = vers la batterie Énergie",
      "orange = pushes towards the Power battery, green = towards the Energy battery"
    ),
  };
}

// Onglet « Décision » : la décision décomposée, et les autres stratégies au même instant.
function OngletDecision({ e, strategie }) {
  const { tr, nombre, pct, partPb, couleurs } = useFormats();
  if (e.demande_nulle) {
    return <Message type="info">{tr("Demande quasi nulle : pas de répartition à décomposer.", "Near-zero demand: no split to break down.")}</Message>;
  }
  const etapes = e.cascade.etapes;
  const valeurs = etapes.map((etape) => etape.valeur * 100);
  const autres = [...e.autres].sort((a, b) => a.alpha - b.alpha); // la plus forte part en haut du graphique
  return (
    <>
      <Graphique
        hauteur={340}
        courbes={[{
          type: "waterfall",
          x: [...etapes.map((etape) => etape.libelle), tr("Décision appliquée", "Decision applied")],
          y: [...valeurs, 0],
          measure: ["absolute", ...etapes.slice(1).map(() => "relative"), "total"],
          text: [
            `${nombre(valeurs[0], 1)} %`, ...valeurs.slice(1).map((v) => `${nombre(v, 1, true)} pts`),
            `${nombre(e.cascade.alpha_final * 100, 1)} %`,
          ],
          textposition: "outside", hoverinfo: "skip",
          connector: { line: { color: COULEURS.secondaire, width: 1 } },
          increasing: { marker: { color: COULEURS.pb } }, decreasing: { marker: { color: COULEURS.eb } },
          totals: { marker: { color: COULEURS.decision } },
        }]}
        disposition={{ showlegend: false, margin: { t: 20, b: 60, l: 56, r: 16 }, yaxis: axe({ title: { text: partPb }, range: [0, 110] }) }}
      />
      <Legende>
        {tr(
          "Part confiée à la batterie Puissance, étape par étape : {l}, violet = décision appliquée.",
          "Share assigned to the Power battery, step by step: {l}, purple = decision applied.",
          { l: couleurs }
        )}
      </Legende>
      {e.cascade.note ? (
        e.cascade.note.type === "alerte" ? <Message type="alerte">{e.cascade.note.texte}</Message> : <Legende>{e.cascade.note.texte}</Legende>
      ) : null}

      <h3>{tr("Les autres stratégies au même instant", "The other strategies at the same time step")}</h3>
      <Graphique
        hauteur={40 * autres.length + 90}
        courbes={[{
          type: "bar", orientation: "h", y: autres.map((a) => a.nom), x: autres.map((a) => a.alpha * 100),
          marker: { color: autres.map((a) => a.couleur), opacity: autres.map((a) => (a.cle === strategie ? 1 : 0.4)) },
          text: autres.map((a) => pct(a.alpha)), textposition: "outside", hoverinfo: "skip", cliponaxis: false,
        }]}
        disposition={{
          showlegend: false, margin: { t: 30, b: 44, l: 8, r: 50 }, xaxis: axe({ title: { text: partPb }, range: [0, 110] }), yaxis: axe(),
          shapes: e.regle_onto
            ? [{ type: "line", yref: "paper", y0: 0, y1: 1, x0: e.regle_onto.alpha * 100, x1: e.regle_onto.alpha * 100,
                 line: { color: COULEURS.reference, dash: "dash", width: 1.5 } }]
            : [],
          annotations: e.regle_onto
            ? [{ x: e.regle_onto.alpha * 100, yref: "paper", y: 1, yanchor: "bottom", showarrow: false,
                 text: tr("règle {i} de l'ontologie", "ontology rule {i}", { i: e.regle_onto.id }), font: { size: 11 } }]
            : [],
        }}
      />
      <Legende>
        {tr("Chaque stratégie décide avec ses propres états de charge à cet instant.", "Each strategy decides with its own states of charge at this time step.")}
      </Legende>
    </>
  );
}

function Contexte({ e }) {
  const { tr } = useFormats();
  const c = e.contexte;
  const courbe = (nom, y, couleur) => ({
    type: "scatter", mode: "lines", x: c.t_s, y, name: nom, line: { color: couleur },
    hovertemplate: `${nom}${tr(" : ", ": ")}%{y:.1f} kW<extra></extra>`,
  });
  return (
    <Depliant titre={tr("Les puissances autour de cet instant", "The powers around this time step")}>
      <Graphique
        hauteur={320}
        courbes={[
          courbe(tr("Demande", "Demand"), c.demande_kw, COULEURS.demande),
          courbe(tr("Batterie Énergie", "Energy battery"), c.eb_kw, COULEURS.eb),
          courbe(tr("Batterie Puissance", "Power battery"), c.pb_kw, COULEURS.pb),
        ]}
        disposition={{
          hovermode: "x unified", xaxis: axe({ title: { text: tr("Temps (s)", "Time (s)") } }),
          yaxis: axe({ title: { text: tr("Puissance (kW)", "Power (kW)") } }),
          shapes: [{ type: "line", yref: "paper", y0: 0, y1: 1, x0: e.instant.t_s, x1: e.instant.t_s,
                     line: { color: COULEURS.reference, dash: "dash", width: 1.5 } }],
        }}
      />
    </Depliant>
  );
}

// Onglet « Raisons » : ce qui a contribué à la décision.
function OngletRaisons({ e }) {
  const { tr, nombre, pct, couleurs } = useFormats();
  const d = e.raisons_detail;
  if (d.type === "vide") {
    return <Message type="info">{tr("Demande quasi nulle : pas de décision à expliquer.", "Near-zero demand: no decision to explain.")}</Message>;
  }
  if (d.type === "regle") {
    return (
      <>
        <p>
          <Texte>{d.texte}</Texte>
        </p>
        <Legende>
          {tr(
            "Le modèle physique ne dépend que de la puissance demandée et du SOC de la batterie Énergie.",
            "The physical model depends only on the power demand and the Energy battery's SOC."
          )}
        </Legende>
      </>
    );
  }
  if (d.type === "regles_floues") {
    return (
      <>
        {d.regles.length ? (
          <Tableau
            colonnes={[
              { cle: "regle", titre: tr("Règle", "Rule") }, { cle: "si", titre: tr("Si", "If") },
              { cle: "verite", titre: tr("Degré de vérité", "Degree of truth"), droite: true },
              { cle: "propose", titre: tr("Propose pour la batterie Puissance", "Proposes for the Power battery"), droite: true },
              { cle: "contribution", titre: tr("Contribution à la décision", "Contribution to the decision"), droite: true },
            ]}
            lignes={d.regles.map((r) => ({
              regle: r.regle, si: r.si, verite: pct(r.verite, 0), propose: pct(r.propose, 0),
              contribution: r.contribution === null ? "—" : `${nombre(r.contribution * 100, 1)} pts`,
            }))}
          />
        ) : null}
        <Legende>
          {tr(
            "La décision des règles est la moyenne de leurs propositions, pondérée par leur degré de vérité : la contribution de chaque règle est donc exacte.",
            "The decision of the rules is the average of their proposals, weighted by their degree of truth: the contribution of each rule is therefore exact."
          )}
        </Legende>
      </>
    );
  }
  const contributions = [...d.contributions].reverse(); // la plus forte en haut
  return (
    <>
      <Graphique
        hauteur={36 * contributions.length + 70}
        courbes={[{
          type: "bar", orientation: "h", y: contributions.map((c) => c.grandeur), x: contributions.map((c) => c.valeur * 100),
          marker: { color: contributions.map((c) => (c.valeur >= 0 ? COULEURS.pb : COULEURS.eb)) },
          text: contributions.map((c) => `${nombre(c.valeur * 100, 1, true)} pts`), textposition: "outside", cliponaxis: false,
          hovertemplate: "%{y} : %{text}<extra></extra>",
        }]}
        disposition={{
          showlegend: false, margin: { t: 10, b: 44, l: 8, r: 60 }, yaxis: axe(),
          xaxis: axe({ title: { text: tr("Contribution à la part de la batterie Puissance (points)", "Contribution to the Power battery's share (points)") } }),
        }}
      />
      <Legende>
        {tr(
          "Par rapport à une situation moyenne du cycle, où le réseau confierait {a} à la batterie Puissance : {l}.",
          "Compared with an average situation of the cycle, where the network would assign {a} to the Power battery: {l}.",
          { a: pct(d.reference), l: couleurs }
        )}
      </Legende>
    </>
  );
}

// Onglet « Connaissances expertes » : l'ontologie, et sa place dans la stratégie.
function OngletConnaissances({ e }) {
  const { tr } = useFormats();
  const c = e.connaissances;
  return (
    <>
      {c.chaine ? (
        <>
          <strong>{c.titre}</strong>
          <Flux etapes={c.chaine} />
        </>
      ) : null}
      {c.etats_transmis ? (
        <p>
          <Texte>{c.etats_transmis}</Texte>
        </p>
      ) : null}
      <h3>{tr("Ce qu'en dit l'ontologie OntoHESS", "What the OntoHESS ontology says")}</h3>
      <p>
        <Texte>{c.etat}</Texte>
      </p>
      <ul>
        {c.regles.map((regle, i) => (
          <li key={i}>
            <Texte>{regle}</Texte>
          </li>
        ))}
      </ul>
      {c.reference ? (
        <>
          <p>
            <Texte>{c.reference}</Texte>
          </p>
          <Legende>
            {tr(
              "Ces règles décrivent la conduite de référence (la batterie Énergie d'abord, dans ses limites) ; s'en écarter n'est pas une erreur. Toutes les règles sont dans « Base de connaissances ».",
              "These rules describe the reference behaviour (the Energy battery first, within its limits); departing from it is not an error. All the rules are in “Knowledge base”."
            )}
          </Legende>
        </>
      ) : null}
      {c.comparaison_ns ? (
        <Depliant titre={tr("Comparer NS-MLP et NS-LSTM au même instant", "Compare NS-MLP and NS-LSTM at the same time step")}>
          <Grille colonnes={2}>
            {c.comparaison_ns.cartes.map((carte) => (
              <Carte key={carte.nom}>
                <h4>{carte.nom}</h4>
                {carte.lignes.map((ligne, i) => (
                  <div key={i}>
                    <Texte>{ligne}</Texte>
                  </div>
                ))}
                <div style={{ marginTop: 8 }}>
                  {carte.verifications.map((v, i) => (
                    <div key={i} className="verification">
                      {v.ok ? "✓" : "✗"} {v.texte}
                    </div>
                  ))}
                </div>
              </Carte>
            ))}
          </Grille>
          <Legende>{c.comparaison_ns.ecart}</Legende>
        </Depliant>
      ) : null}
    </>
  );
}

// Schéma du HESS, chaque composant coloré selon son poids dans la décision du GNN.
function SchemaGnn({ schema }) {
  const { nombre } = useLangue();
  const positions = [[0, 1], [0, -1], [1.2, 0], [2.4, 0], [3.6, 0]];
  const lx = [];
  const ly = [];
  schema.liaisons.forEach(([a, b]) => {
    if (a < positions.length && b < positions.length) {
      lx.push(positions[a][0], positions[b][0], null);
      ly.push(positions[a][1], positions[b][1], null);
    }
  });
  const poids = schema.composants.map((c) => c.poids);
  return (
    <Graphique
      hauteur={360}
      courbes={[
        { type: "scatter", mode: "lines", x: lx, y: ly, line: { color: "#C7CCD6", width: 2 }, hoverinfo: "skip" },
        {
          type: "scatter", mode: "markers+text", x: positions.map((p) => p[0]), y: positions.map((p) => p[1]),
          marker: { size: poids.map((p) => 34 + p * 0.9), color: poids, colorscale: "Blues", showscale: true, colorbar: { title: { text: "%" } }, line: { color: "white", width: 2 } },
          text: schema.composants.map((c) => `${c.nom}<br>${nombre(c.poids, 0)} %`), textposition: "bottom center", hoverinfo: "text",
        },
      ]}
      disposition={{ showlegend: false, margin: { t: 20, b: 20, l: 10, r: 10 }, xaxis: { visible: false }, yaxis: { visible: false, range: [-1.8, 1.6] } }}
    />
  );
}

// Onglet « Réseau de neurones » : la part apprise de la décision.
function OngletReseau({ e }) {
  const { tr, nombre, pct } = useFormats();
  const r = e.reseau;
  if (r.type === "aucun") {
    return (
      <Message type="info">
        {tr(
          "Cette stratégie n'a pas de réseau de neurones : sa décision est entièrement lisible (onglets Raisons et Connaissances expertes).",
          "This strategy has no neural network: its decision can be read in full (Reasons and Expert knowledge tabs)."
        )}
      </Message>
    );
  }
  if (r.type === "vide") {
    return <Message type="info">{tr("Demande quasi nulle : pas de décision à expliquer.", "Near-zero demand: no decision to explain.")}</Message>;
  }
  if (r.type === "ns_mlp") {
    return (
      <>
        <Grille colonnes={3}>
          <Indicateur titre={tr("Correction du réseau", "Network correction")} valeur={`${nombre(r.correction * 100, 1, true)} pts`} />
          <Indicateur titre={tr("Part de la correction permise", "Share of the allowed correction")} valeur={pct(r.part_permise, 0)} />
          <Indicateur
            titre={tr("Garde-fou de l'ontologie", "Ontology safeguard")}
            valeur={r.garde_fou ? tr("intervient", "steps in") : tr("n'intervient pas", "does not step in")}
          />
        </Grille>
        <p>
          {tr(
            "Le réseau ne décide pas seul : il ajoute aux règles floues une correction limitée à ±{m} points. Le calcul de cette correction n'est pas lisible, mais son poids dans la décision est connu exactement. Quand le SOC de la batterie Puissance passe sous {r} %, un garde-fou de l'ontologie (règles R14 et R16) limite en traction sa part à ce que la batterie Énergie ne peut pas fournir.",
            "The network does not decide alone: it adds to the fuzzy rules a correction limited to ±{m} points. The calculation of this correction cannot be read, but its weight in the decision is known exactly. When the Power battery's SOC drops below {r} %, an ontology safeguard (rules R14 and R16) limits its share in traction to what the Energy battery cannot supply.",
            { m: nombre(r.correction_max * 100, 0), r: nombre(r.reserve * 100, 0) }
          )}
        </p>
      </>
    );
  }
  return (
    <>
      <p>
        {tr(
          "Comment l'explication est reconstruite : on remplace tour à tour chacune des {k} grandeurs d'entrée par sa valeur moyenne sur le cycle, dans toutes les combinaisons possibles, et on mesure de combien la décision change (méthode de Shapley). La situation moyenne plus les contributions redonnent exactement la décision du réseau ({a} + contributions = {b}).",
          "How the explanation is reconstructed: each of the {k} input quantities is replaced in turn by its average value over the cycle, in all possible combinations, and we measure how much the decision changes (Shapley method). The average situation plus the contributions give back exactly the network's decision ({a} + contributions = {b}).",
          { k: r.nb_grandeurs, a: pct(r.reference), b: pct(r.alpha_explique) }
        )}
      </p>
      {r.memoire_s ? (
        <Legende>
          {tr(
            "Chaque grandeur compte pour toute son évolution sur les {w} dernières secondes.",
            "Each quantity counts for its whole evolution over the last {w} seconds.",
            { w: r.memoire_s }
          )}
        </Legende>
      ) : null}
      {r.poids_ontologie !== null ? (
        <Grille colonnes={3}>
          <Indicateur
            titre={tr("Poids des états déduits par l'ontologie", "Weight of the states inferred by the ontology")}
            valeur={pct(r.poids_ontologie, 0)}
          />
        </Grille>
      ) : null}
      {r.schema ? (
        <>
          <p>
            <Texte>
              {tr(
                "**Poids de chaque composant du HESS dans la décision** (sensibilité de la décision à ses grandeurs)",
                "**Weight of each HESS component in the decision** (sensitivity of the decision to its quantities)"
              )}
            </Texte>
          </p>
          <SchemaGnn schema={r.schema} />
        </>
      ) : null}
    </>
  );
}

// Onglet « Et si… ? » : que déciderait la stratégie si la situation changeait un peu ?
function OngletEtSi({ e }) {
  const { tr, nombre, kw, pct } = useFormats();
  if (!e.et_si) {
    return (
      <Message type="info">
        {tr("Demande quasi nulle : choisissez un instant de traction ou de freinage.", "Near-zero demand: choose a time step in traction or braking.")}
      </Message>
    );
  }
  const s = e.et_si;
  return (
    <>
      <p>
        <Texte>
          {tr(
            "Que déciderait **{s}** si la situation changeait un peu ? Ici : **{a}** pour la batterie Puissance ({p}).",
            "What would **{s}** decide if the situation changed slightly? Here: **{a}** for the Power battery ({p}).",
            { s: e.strategie.nom, a: pct(s.base.alpha), p: kw(s.base.p_w) }
          )}
        </Texte>
      </p>
      <Tableau
        colonnes={[
          { cle: "libelle", titre: tr("Variante", "Variant") },
          { cle: "part", titre: tr("Part de la batterie Puissance", "Power battery share"), droite: true },
          { cle: "puissance", titre: tr("Puissance de la batterie Puissance", "Power battery power"), droite: true },
          { cle: "variation", titre: tr("Variation de cette puissance", "Change in that power"), droite: true },
          { cle: "conforme", titre: tr("Sens conforme à la physique", "Direction consistent with physics"), droite: true },
        ]}
        lignes={s.variantes.map((v) => ({
          libelle: v.libelle, part: pct(v.alpha), puissance: kw(v.p_w),
          variation: `${nombre(v.variation_w / 1000, 2, true)} kW`, conforme: v.conforme ? "✓" : "✗",
        }))}
      />
      <Legende>
        {tr(
          "Sens attendu : plus de charge dans la batterie Puissance, ou plus de demande, ne doit pas réduire la puissance qu'elle fournit ; plus de charge dans la batterie Énergie ne doit pas l'augmenter (à 1 % de la demande près). C'est le test de cohérence physique E3, appliqué à cet instant.",
          "Expected direction: more charge in the Power battery, or more demand, must not reduce the power it supplies; more charge in the Energy battery must not increase it (to within 1 % of the demand). This is the E3 physical-consistency test, applied at this time step."
        )}
      </Legende>
      <h3>{tr("D'après les seuils de l'ontologie", "According to the ontology's thresholds")}</h3>
      <ul>
        {s.contrefactuels.map((phrase, i) => (
          <li key={i}>{phrase}</li>
        ))}
      </ul>
      {s.regles_non_appliquees.length ? (
        <Depliant titre={tr("Règles de l'ontologie qui ne s'appliquent pas, et pourquoi", "Ontology rules that do not apply, and why")}>
          <ul>
            {s.regles_non_appliquees.map((regle, i) => (
              <li key={i}>
                <Texte>{regle}</Texte>
              </li>
            ))}
          </ul>
        </Depliant>
      ) : null}
    </>
  );
}

// Onglet « Sur tout le cycle » : bilan de l'explicabilité.
function OngletCycle({ cycle, strategie, nom }) {
  const { tr, nombre, pct, partPb } = useFormats();
  const bilan = useDonnees(`/cycles/${cycle}/explication-cycle`, { strategie });
  return (
    <Attente etat={bilan} texte={tr("Analyse du cycle…", "Analysing the cycle…")}>
      {(b) => {
        const f = b.au_fil_du_cycle;
        const courbes = [
          { type: "scatter", mode: "lines", x: f.temps_min, y: f.decision_pct, name: tr("Décision de {s}", "Decision of {s}", { s: nom }),
            line: { color: COULEURS.decision, width: 2.2 } },
          { type: "scatter", mode: "lines", x: f.temps_min, y: f.ontologie_pct, name: tr("Règle de l'ontologie", "Ontology rule"),
            line: { color: COULEURS.reference, width: 1.6, dash: "dash" } },
        ];
        if (f.regles_pct) {
          courbes.push({ type: "scatter", mode: "lines", x: f.temps_min, y: f.regles_pct, name: tr("Règles floues", "Fuzzy rules"),
                         line: { color: COULEURS.secondaire, width: 1.6, dash: "dot" } });
        }
        const dominantes = b.regles ? [...b.regles.dominantes].reverse() : [];
        return (
          <>
            <Grille colonnes={4}>
              <Indicateur titre={tr("Explication", "Explanation")} valeur={b.exacte ? tr("exacte", "exact") : tr("reconstruite", "reconstructed")} />
              <Indicateur titre={tr("Cohérence physique (E3)", "Physical consistency (E3)")} valeur={pct(b.e3, 0)} />
              <Indicateur titre={tr("Accord avec la règle de l'ontologie", "Agreement with the ontology rule")} valeur={b.accord_ontologie === null ? "—" : pct(b.accord_ontologie, 0)} />
              <Indicateur titre={tr("Décisions corrigées par le filtre", "Decisions corrected by the filter")} valeur={b.part_corrigee === null ? "—" : pct(b.part_corrigee)} />
            </Grille>
            <Legende>{b.legende}</Legende>
            {b.regles ? (
              <Grille colonnes={3}>
                <Indicateur titre={tr("Règles floues vraies par décision", "Fuzzy rules true per decision")} valeur={nombre(b.regles.vraies_par_decision, 1)} />
                {b.regles.correction_utilisee !== null ? (
                  <Indicateur titre={tr("Part de la correction permise utilisée", "Share of the allowed correction used")} valeur={pct(b.regles.correction_utilisee, 0)} />
                ) : null}
                {b.regles.garde_fou !== null ? (
                  <Indicateur
                    titre={tr("Décisions reprises par le garde-fou", "Decisions taken over by the safeguard")} valeur={pct(b.regles.garde_fou)}
                    aide={tr("Instants de traction où la batterie Puissance était sous {r} % de SOC.", "Traction time steps where the Power battery was below {r} % SOC.", { r: nombre(b.regles.reserve * 100, 0) })}
                  />
                ) : null}
                {b.regles.regles_seules !== null ? (
                  <Indicateur titre={tr("Décisions laissées aux règles seules", "Decisions left to the rules alone")} valeur={pct(b.regles.regles_seules)} />
                ) : null}
              </Grille>
            ) : null}
            <h3>{tr("Décision et règle de référence au fil du cycle", "Decision and reference rule over the cycle")}</h3>
            <Graphique
              hauteur={340} courbes={courbes}
              disposition={{
                hovermode: "x unified", xaxis: axe({ title: { text: tr("Temps (min)", "Time (min)") } }),
                yaxis: axe({ title: { text: `${partPb} — ${tr("moyenne par minute", "mean per minute")}` } }),
              }}
            />
            {b.regles ? (
              <>
                <h3>{tr("Règles floues qui pèsent le plus sur le cycle", "Fuzzy rules that weigh most over the cycle")}</h3>
                <Graphique
                  hauteur={36 * dominantes.length + 70}
                  courbes={[{
                    type: "bar", orientation: "h", y: dominantes.map((d) => d.regle), x: dominantes.map((d) => d.part * 100),
                    marker: { color: COULEURS.secondaire }, text: dominantes.map((d) => pct(d.part, 0)), textposition: "outside", cliponaxis: false,
                    hovertemplate: "%{y} : %{text}<extra></extra>",
                  }]}
                  disposition={{
                    showlegend: false, margin: { t: 10, b: 44, l: 8, r: 60 }, yaxis: axe(),
                    xaxis: axe({ title: { text: tr("Part des instants où la règle pèse le plus (%)", "Share of time steps where the rule weighs most (%)") } }),
                  }}
                />
              </>
            ) : null}
          </>
        );
      }}
    </Attente>
  );
}

export default function Explication() {
  const { tr, nombre, kw, pct } = useFormats();
  const { cycle, instant, setInstant } = useEtat();
  const infos = useCycle();
  const [strategie, setStrategie] = useStrategie(infos.donnees);

  // Instant étudié, en secondes : par défaut celui que propose le serveur (une demande franche
  // près du milieu du cycle), ramené dans les bornes du cycle.
  const bornes = infos.donnees ? [Math.ceil(infos.donnees.t_min), Math.floor(infos.donnees.t_max)] : null;
  const t = bornes ? Math.min(Math.max(instant ?? Math.round(infos.donnees.t_defaut), bornes[0]), bornes[1]) : null;
  const [demande, setDemande] = useState(t); // suit le curseur avec un léger délai
  useEffect(() => {
    const minuteur = setTimeout(() => setDemande(t), 250);
    return () => clearTimeout(minuteur);
  }, [t]);

  const pret = strategie && demande !== null;
  const explication = useDonnees(pret ? `/cycles/${cycle}/explication` : null, { strategie, t: demande }, { conserver: true });

  return (
    <>
      <h1>{tr("💡 Pourquoi cette décision ?", "💡 Why this decision?")}</h1>
      <Legende>
        {tr(
          "Pour un instant du cycle et une stratégie : quelle décision a été prise, et pourquoi. L'explication est exacte quand le calcul de la stratégie est lisible, reconstruite après coup quand il ne l'est pas.",
          "For one time step of the cycle and one strategy: which decision was made, and why. The explanation is exact when the strategy's calculation can be read, reconstructed afterwards when it cannot."
        )}
      </Legende>
      <div className="rangee">
        {bornes ? (
          <div style={{ flex: 2 }}>
            <Curseur titre={tr("Instant du cycle (s)", "Time in the cycle (s)")} valeur={t} onChange={setInstant} min={bornes[0]} max={bornes[1]} unite=" s" />
          </div>
        ) : null}
        <SelecteurStrategie infos={infos.donnees} valeur={strategie} onChange={setStrategie} />
        <SelecteurCycle />
      </div>

      <Attente etat={explication} texte={tr("Calcul de l'explication…", "Computing the explanation…")}>
        {(e) => (
          <>
            <Carte>
              <p>
                <Texte>
                  {tr(
                    "**{s}** · instant t = {t} s · vitesse {v} km/h · état déduit par l'ontologie : {e}",
                    "**{s}** · time t = {t} s · speed {v} km/h · state inferred by the ontology: {e}",
                    { s: e.strategie.nom, t: nombre(e.instant.t_s, 0), v: nombre(e.instant.vitesse_kmh, 0), e: e.instant.etat.charAt(0).toLowerCase() + e.instant.etat.slice(1) }
                  )}
                </Texte>
              </p>
              <Grille colonnes={6}>
                <Indicateur titre={tr("Demande", "Demand")} valeur={kw(e.decision.demande_w)} />
                <Indicateur titre="SOC EB" valeur={pct(e.decision.soc_eb)} />
                <Indicateur titre="SOC PB" valeur={pct(e.decision.soc_pb)} />
                <Indicateur titre={tr("alpha (part de la PB)", "alpha (PB share)")} valeur={pct(e.decision.alpha)} />
                <Indicateur titre={tr("Batterie Énergie", "Energy battery")} valeur={kw(e.decision.p_eb_w)} />
                <Indicateur titre={tr("Batterie Puissance", "Power battery")} valeur={kw(e.decision.p_pb_w)} />
              </Grille>
            </Carte>

            <h2>{tr("Pourquoi cette décision ?", "Why this decision?")}</h2>
            <ol className="raisons">
              {e.raisons.map((raison, i) => (
                <li key={i}>
                  <Texte>{raison}</Texte>
                </li>
              ))}
            </ol>
            <Legende>
              {e.nature.exacte ? tr("Explication exacte : ", "Exact explanation: ") : tr("Explication reconstruite : ", "Reconstructed explanation: ")}
              {e.nature.texte}
            </Legende>

            <Onglets
              onglets={[
                { titre: tr("Décision", "Decision"), contenu: () => (<><OngletDecision e={e} strategie={strategie} /><Contexte e={e} /></>) },
                { titre: tr("Raisons", "Reasons"), contenu: () => <OngletRaisons e={e} /> },
                { titre: tr("Connaissances expertes", "Expert knowledge"), contenu: () => <OngletConnaissances e={e} /> },
                { titre: tr("Réseau de neurones", "Neural network"), contenu: () => <OngletReseau e={e} /> },
                { titre: tr("Et si… ?", "What if…?"), contenu: () => <OngletEtSi e={e} /> },
                { titre: tr("Sur tout le cycle", "Over the whole cycle"), contenu: () => <OngletCycle cycle={cycle} strategie={strategie} nom={e.strategie.nom} /> },
              ]}
            />
          </>
        )}
      </Attente>

      <PiedNavigation page="/explication" />
    </>
  );
}
