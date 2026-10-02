// Base de connaissances : l'ontologie OntoHESS dans un ordre pédagogique — à quoi
// elle sert, ce qu'elle décrit, ses règles en langage courant, son rôle dans chaque
// stratégie — puis le test de ses règles sur une situation choisie.

import { useEffect, useState } from "react";
import { useDonnees } from "../api.js";
import {
  Attente, Carte, Curseur, Depliant, Flux, Graphique, Grille, Indicateur, Legende, Message, Onglets, PiedNavigation,
  Radios, Selection, Tableau, Texte,
} from "../composants.jsx";
import { COULEURS } from "../couleurs.js";
import { useLangue } from "../langue.jsx";

// Objets décrits par l'ontologie et leurs relations ; l'état déduit est mis en avant.
function GrapheConnaissances({ etats, etatActif }) {
  const { tr } = useLangue();
  const positionsEtats = { state_Normal: -1.6, state_Overload_High: 0, state_Overload_Low: 1.6 };
  const noeuds = {
    hess1: [0, 2, tr("Système HESS", "HESS system")],
    managementStrategy1: [-2.2, 2, tr("Stratégie de gestion", "Management strategy")],
    batteryE1: [-1.6, 1, tr("Batterie Énergie", "Energy battery")],
    batteryP1: [0, 1, tr("Batterie Puissance", "Power battery")],
    converter1: [1.6, 1, tr("Convertisseur", "Converter")],
    load1: [1.6, 0, tr("Charge (moteur)", "Load (motor)")],
  };
  etats.forEach((e) => {
    noeuds[e.cle] = [positionsEtats[e.cle], -1, e.libelle];
  });
  const liaisons = [
    ["hess1", "managementStrategy1"], ["hess1", "batteryE1"], ["hess1", "batteryP1"], ["hess1", "converter1"],
    ["converter1", "batteryE1"], ["converter1", "load1"], ["hess1", "state_Normal"], ["hess1", "state_Overload_High"],
    ["hess1", "state_Overload_Low"],
  ];
  const cles = Object.keys(noeuds);
  const estEtat = (cle) => cle.startsWith("state_");
  const courbes = liaisons.map(([a, b]) => ({
    type: "scatter", mode: "lines", x: [noeuds[a][0], noeuds[b][0]], y: [noeuds[a][1], noeuds[b][1]],
    line: { color: b === etatActif ? COULEURS.decision : "#9AA0AA", width: b === etatActif ? 3 : 1.2 },
    hoverinfo: "skip", showlegend: false,
  }));
  courbes.push({
    type: "scatter", mode: "markers+text", x: cles.map((c) => noeuds[c][0]), y: cles.map((c) => noeuds[c][1]),
    marker: {
      size: cles.map((c) => (c === etatActif ? 38 : estEtat(c) ? 24 : 30)),
      color: cles.map((c) => (c === etatActif ? COULEURS.decision : estEtat(c) ? COULEURS.secondaire : COULEURS.reference)),
      line: { color: "white", width: 2 },
    },
    text: cles.map((c) => noeuds[c][2]), textposition: "bottom center", textfont: { size: 11 },
    hovertext: cles.map((c) => tr("{l} (nom dans l'ontologie : {c})", "{l} (name in the ontology: {c})", { l: noeuds[c][2], c })),
    hoverinfo: "text", showlegend: false,
  });
  return (
    <Graphique
      hauteur={380} courbes={courbes}
      disposition={{ margin: { t: 10, b: 10, l: 10, r: 10 }, xaxis: { visible: false, range: [-3, 2.8] }, yaxis: { visible: false, range: [-1.9, 2.5] } }}
    />
  );
}

function TesterLesRegles() {
  const { tr } = useLangue();
  const [situation, setSituation] = useState({ p: 15, eb: 60, pb: 80 });
  const [demande, setDemande] = useState(situation); // suit les curseurs avec un léger délai
  useEffect(() => {
    const minuteur = setTimeout(() => setDemande(situation), 200);
    return () => clearTimeout(minuteur);
  }, [situation]);
  const test = useDonnees("/ontologie/test", { p_kw: demande.p, soc_eb: demande.eb / 100, soc_pb: demande.pb / 100 });
  const regler = (cle) => (valeur) => setSituation((s) => ({ ...s, [cle]: valeur }));

  return (
    <>
      <h2>{tr("Tester les règles sur une situation", "Test the rules on a situation")}</h2>
      <Legende>
        {tr(
          "Choisissez une situation : l'ontologie en déduit l'état de fonctionnement et la répartition de référence, règle par règle.",
          "Choose a situation: the ontology infers the operating state and the reference split, rule by rule."
        )}
      </Legende>
      <Grille colonnes={3}>
        <Curseur titre={tr("Puissance demandée (kW)", "Power demand (kW)")} valeur={situation.p} onChange={regler("p")} min={-50} max={50} pas={0.5} />
        <Curseur titre={tr("SOC de la batterie Énergie (%)", "Energy battery SOC (%)")} valeur={situation.eb} onChange={regler("eb")} min={0} max={100} />
        <Curseur titre={tr("SOC de la batterie Puissance (%)", "Power battery SOC (%)")} valeur={situation.pb} onChange={regler("pb")} min={0} max={100} />
      </Grille>
      <Attente etat={test}>
        {(t) => (
          <>
            <Grille colonnes={2}>
              <GrapheConnaissances etats={t.etats} etatActif={t.etat.cle} />
              <div>
                <p>
                  {tr("État déduit : ", "Inferred state: ")}
                  <strong>{t.etat.libelle}</strong>
                </p>
                <ul>
                  {t.regles.map((regle, i) => (
                    <li key={i}>
                      <Texte>{regle}</Texte>
                    </li>
                  ))}
                </ul>
                <Message type={t.conclusion.type}>{t.conclusion.texte}</Message>
              </div>
            </Grille>
            <Depliant titre={tr("Règles qui ne s'appliquent pas, et pourquoi", "Rules that do not apply, and why")}>
              <ul>
                {t.non_appliquees.map((regle, i) => (
                  <li key={i}>
                    <Texte>{regle}</Texte>
                  </li>
                ))}
              </ul>
            </Depliant>
          </>
        )}
      </Attente>
    </>
  );
}

function ParcourirRegles({ o }) {
  const { tr } = useLangue();
  const [role, setRole] = useState("toutes");
  const choix = o.regles.filter((r) => role === "toutes" || r.type === role);
  const [idRegle, setIdRegle] = useState(choix[0].id);
  const regle = choix.find((r) => r.id === idRegle) || choix[0];
  return (
    <>
      <Radios
        titre={tr("Rôle de la règle", "Role of the rule")} valeur={role} onChange={setRole}
        options={[{ valeur: "toutes", nom: tr("Toutes", "All") }, ...o.roles.map((r) => ({ valeur: r.cle, nom: r.nom }))]}
      />
      <Selection
        titre={tr("Règle", "Rule")} valeur={regle.id} onChange={setIdRegle}
        options={choix.map((r) => ({ valeur: r.id, nom: `${r.id} — ${r.lecture}` }))}
      />
      <Carte>
        <h4>
          {regle.id} · {regle.role}
        </h4>
        <p>
          <strong>{tr("En clair", "In plain words")}</strong>
          {tr(" : ", ": ")}
          <Texte>{regle.phrase}</Texte>
        </p>
        <Legende>{tr("Écriture de la règle dans l'ontologie :", "How the rule is written in the ontology:")}</Legende>
        <Grille colonnes={2}>
          <div>
            <strong>{tr("S'applique à", "Applies to")}</strong>
            <p>{regle.sapplique_a}</p>
            <strong>{tr("Grandeurs lues", "Quantities read")}</strong>
            <Texte>{regle.grandeurs_lues.length ? regle.grandeurs_lues.map((g) => `- ${g}`).join("\n") + "\n" : "—"}</Texte>
          </div>
          <div>
            <strong>{tr("Si", "If")}</strong>
            <Texte>
              {regle.conditions.length ? regle.conditions.map((c) => `- ${c}`).join("\n") + "\n" : tr("- toujours (pas de condition)\n", "- always (no condition)\n")}
            </Texte>
            {regle.calculs.length ? (
              <>
                <strong>{tr("Calcule", "Computes")}</strong>
                <Texte>{regle.calculs.map((c) => `- ${c}`).join("\n") + "\n"}</Texte>
              </>
            ) : null}
            <strong>{tr("Alors", "Then")}</strong>
            <Texte>{regle.conclusions.length ? regle.conclusions.map((c) => `- ${c}`).join("\n") + "\n" : "—"}</Texte>
          </div>
        </Grille>
        <Legende>
          {regle.appliquee
            ? tr("Règle appliquée à chaque instant du cycle.", "Rule applied at each time step of the cycle.")
            : tr(
                "Règle non appliquée pendant la simulation : elle porte sur des courants et tensions internes que la simulation ne calcule pas, ou n'a pas de condition.",
                "Rule not applied during the simulation: it concerns internal currents and voltages that the simulation does not compute, or has no condition."
              )}
        </Legende>
      </Carte>
      <Depliant titre={tr("Vue d'ensemble des {n} règles", "Overview of the {n} rules", { n: o.regles.length })}>
        <Tableau
          hauteur={420}
          colonnes={[
            { cle: "id", titre: tr("Règle", "Rule") }, { cle: "role", titre: tr("Rôle", "Role") },
            { cle: "si", titre: tr("Si", "If") }, { cle: "alors", titre: tr("Alors", "Then") },
          ]}
          lignes={o.regles.map((r) => ({
            id: r.id, role: r.role, si: r.conditions.join(tr(" et ", " and ")) || tr("toujours", "always"), alors: r.lecture,
          }))}
        />
      </Depliant>
      <Depliant titre={tr("Signification des symboles", "Meaning of the symbols")}>
        <ul>
          {o.glossaire.map((g) => (
            <li key={g.symbole}>
              <code>{g.symbole}</code> : {g.sens}
            </li>
          ))}
        </ul>
      </Depliant>
    </>
  );
}

function ReglesFloues({ o }) {
  const { tr, nombre } = useLangue();
  const [cle, setCle] = useState(o.regles_floues[0].cle);
  const regle = o.regles_floues.find((r) => r.cle === cle) || o.regles_floues[0];
  return (
    <>
      <Legende>
        {tr(
          "Ces règles sont propres à la stratégie « logique floue » : elles ne figurent pas dans le fichier de l'ontologie, mais emploient ses concepts. Une condition y est vraie à un certain degré, entre 0 et 1 ; la décision est la moyenne des conclusions des règles, pondérée par ces degrés. La contribution de chaque règle à la décision est donc connue exactement.",
          "These rules belong to the “fuzzy logic” strategy: they are not in the ontology file, but use its concepts. A condition is true to a certain degree, between 0 and 1; the decision is the average of the rules' conclusions, weighted by these degrees. The contribution of each rule to the decision is therefore known exactly."
        )}
      </Legende>
      <Selection titre={tr("Règle floue", "Fuzzy rule")} valeur={regle.cle} onChange={setCle} options={o.regles_floues.map((r) => ({ valeur: r.cle, nom: r.libelle }))} />
      <Carte>
        <h4>{regle.libelle}</h4>
        <p>
          <strong>{tr("Si", "If")}</strong> {regle.si}
        </p>
        <p>
          <Texte>
            {tr(
              "**Alors** confier **{a} %** de la puissance à la batterie Puissance, c'est-à-dire {s}.",
              "**Then** assign **{a} %** of the power to the Power battery, i.e. {s}.",
              { a: nombre(regle.alpha * 100, 0), s: regle.sens }
            )}
          </Texte>
        </p>
        {regle.concepts.length ? (
          <p>
            <strong>{tr("Concepts de l'ontologie", "Ontology concepts")}</strong>
            {tr(" : ", ": ")}
            {regle.concepts.join(", ")}
          </p>
        ) : null}
      </Carte>
      <Tableau
        colonnes={[
          { cle: "libelle", titre: tr("Règle", "Rule") }, { cle: "si", titre: tr("Si", "If") },
          { cle: "part", titre: tr("Part de la batterie Puissance", "Power battery share"), droite: true },
        ]}
        lignes={o.regles_floues.map((r) => ({ libelle: r.libelle, si: r.si, part: `${nombre(r.alpha * 100, 0)} %` }))}
      />
      <Depliant titre={tr("Définition chiffrée des conditions", "Numerical definition of the conditions")}>
        <ul>
          {o.termes_flous.map((t) => (
            <li key={t.terme}>
              <strong>{t.terme}</strong> : {t.definition}
            </li>
          ))}
        </ul>
        <Legende>
          {tr(
            "Quand aucune règle n'est vraie, la répartition par défaut est de {a} % pour la batterie Puissance.",
            "When no rule is true, the default split is {a} % for the Power battery.",
            { a: nombre(o.alpha_par_defaut * 100, 0) }
          )}
        </Legende>
      </Depliant>
    </>
  );
}

export default function Connaissances() {
  const { tr } = useLangue();
  const ontologie = useDonnees("/ontologie");

  return (
    <>
      <h1>{tr("📚 Base de connaissances", "📚 Knowledge base")}</h1>
      <h3>{tr("OntoHESS : les connaissances sur le HESS, écrites noir sur blanc", "OntoHESS: knowledge about the HESS, written down explicitly")}</h3>
      <Legende>
        {tr(
          "Une ontologie est une description structurée d'un domaine : ici, les composants du HESS, leurs grandeurs, leurs états et les règles d'un expert pour interpréter la situation du système. La page « Pourquoi cette décision ? » l'applique à chaque instant du cycle.",
          "An ontology is a structured description of a domain: here, the components of the HESS, their quantities, their states and an expert's rules for interpreting the situation of the system. The “Why this decision?” page applies it at each time step of the cycle."
        )}
      </Legende>

      <Attente etat={ontologie}>
        {(o) =>
          !o.disponible ? (
            <Message type="alerte">{tr("L'ontologie n'a pas pu être lue.", "The ontology could not be read.")}</Message>
          ) : (
            <>
              <p>
                <strong>{tr("Contenu de l'ontologie", "Contents of the ontology")}</strong>
              </p>
              <Grille colonnes={5}>
                <Indicateur titre={tr("Concepts", "Concepts")} valeur={o.compteurs.concepts}
                  aide={tr("Les notions décrites : batterie, convertisseur, seuil, état…", "The notions described: battery, converter, threshold, state…")} />
                <Indicateur titre={tr("Relations", "Relations")} valeur={o.compteurs.relations}
                  aide={tr("Les liens entre concepts : « est composé de », « alimente »…", "The links between concepts: “is composed of”, “powers”…")} />
                <Indicateur titre={tr("Propriétés", "Properties")} valeur={o.compteurs.proprietes}
                  aide={tr("Les grandeurs attachées aux concepts : tension, SOC, puissance maximale…", "The quantities attached to concepts: voltage, SOC, maximum power…")} />
                <Indicateur titre={tr("Objets décrits", "Objects described")} valeur={o.compteurs.objets}
                  aide={tr("Les éléments concrets du HESS étudié, avec leurs valeurs", "The actual elements of the HESS studied, with their values")} />
                <Indicateur titre={tr("Règles", "Rules")} valeur={o.compteurs.regles}
                  aide={tr("Règles logiques du type « si … alors … »", "Logical rules of the form “if … then …”")} />
              </Grille>
              <Flux
                etapes={[
                  { texte: tr("Situation du HESS", "Situation of the HESS"), role: "reference" },
                  { texte: tr("Grandeurs électriques + SOC", "Electrical quantities + SOC"), role: "reference" },
                  { texte: tr("État du système", "System state"), role: "reference" },
                  { texte: tr("Règles expertes", "Expert rules"), role: "secondaire" },
                  { texte: tr("État déduit", "Inferred state"), role: "secondaire" },
                  { texte: tr("Décision de gestion", "Management decision"), role: "decision" },
                ]}
              />
              <Legende>
                {tr("Le chemin que décrit l'ontologie : de la situation mesurée à la décision de gestion.", "The path described by the ontology: from the measured situation to the management decision.")}
              </Legende>

              <h2>{tr("A. À quoi sert OntoHESS ?", "A. What is OntoHESS for?")}</h2>
              <Texte>
                {tr(
                  "- **Décrire** le système avec un vocabulaire commun : batteries, convertisseur, grandeurs, seuils.\n- **Raisonner** : à partir de la puissance demandée et des SOC, ses règles en déduisent l'état de fonctionnement et une répartition de référence de la puissance.\n- **Expliquer** : chaque décision d'une stratégie peut être relue avec ces règles, et les stratégies neuro-symboliques s'appuient directement sur ses concepts.",
                  "- **Describe** the system with a shared vocabulary: batteries, converter, quantities, thresholds.\n- **Reason**: from the power demand and the SOCs, its rules infer the operating state and a reference power split.\n- **Explain**: each decision of a strategy can be re-read with these rules, and the neuro-symbolic strategies rely directly on its concepts."
                )}
              </Texte>

              <h2>{tr("B. Que décrit OntoHESS ?", "B. What does OntoHESS describe?")}</h2>
              <Grille colonnes={o.categories.length}>
                {o.categories.map((categorie) => (
                  <Carte key={categorie.cle}>
                    <strong>{categorie.nom}</strong>
                    <Legende>{categorie.description}</Legende>
                    <div>{tr("{n} concepts", "{n} concepts", { n: categorie.concepts.length })}</div>
                  </Carte>
                ))}
              </Grille>
              <Depliant titre={tr("Détails pour les spécialistes de l'ontologie", "Details for ontology specialists")}>
                <Legende>{tr("Les noms ci-dessous sont ceux du fichier de l'ontologie (en anglais).", "The names below are those of the ontology file.")}</Legende>
                <p>
                  <Texte>{tr("**Concepts**, rangés sous leurs quatre grandes catégories", "**Concepts**, grouped under their four main categories")}</Texte>
                </p>
                <Grille colonnes={o.categories.length}>
                  {o.categories.map((categorie) => (
                    <div key={categorie.cle}>
                      <code>{categorie.cle}</code>
                      <ul>
                        {categorie.concepts.map((c) => (
                          <li key={c.nom}>
                            <code>{c.nom}</code>
                            {c.clair ? ` — ${c.clair}` : ""}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </Grille>
                <p>
                  <Texte>
                    {tr(
                      "**Objets décrits** : les éléments concrets du HESS étudié, avec les valeurs que l'ontologie leur attribue",
                      "**Objects described**: the actual elements of the HESS studied, with the values the ontology gives them"
                    )}
                  </Texte>
                </p>
                <Tableau
                  colonnes={[
                    { cle: "nom", titre: tr("Objet", "Object") }, { cle: "concepts", titre: tr("Concepts", "Concepts") },
                    { cle: "proprietes", titre: tr("Propriétés", "Properties") },
                  ]}
                  lignes={o.objets}
                />
                <Message type="alerte">{o.avertissement}</Message>
                <p>
                  <Texte>{tr("**Relations** entre concepts ({n})", "**Relations** between concepts ({n})", { n: o.relations.length })}</Texte>
                </p>
                <Tableau
                  hauteur={360}
                  colonnes={[{ cle: "nom", titre: tr("Relation", "Relation") }, { cle: "de", titre: tr("De", "From") }, { cle: "vers", titre: tr("Vers", "To") }]}
                  lignes={o.relations}
                />
                <p>
                  <Texte>{tr("**Propriétés** ({n})", "**Properties** ({n})", { n: o.proprietes.length })}</Texte>
                </p>
                <Legende>{o.proprietes.map((p) => `\`${p}\``).join(" · ")}</Legende>
              </Depliant>

              <h2>{tr("C. Quelles règles sont utilisées ?", "C. Which rules are used?")}</h2>
              <Legende>
                {tr(
                  "Les règles lues à chaque instant du cycle, en langage courant. Elles reconnaissent d'abord le mode de fonctionnement, puis prescrivent une répartition de la puissance.",
                  "The rules read at each time step of the cycle, in plain language. They first recognise the operating mode, then prescribe a power split."
                )}
              </Legende>
              <Grille colonnes={2}>
                {[
                  ["mode", tr("Reconnaître le mode de fonctionnement", "Recognise the operating mode")],
                  ["repartition", tr("Répartir la puissance entre les batteries", "Split the power between the batteries")],
                ].map(([type, titre]) => (
                  <Carte key={type}>
                    <strong>{titre}</strong>
                    <ul>
                      {o.regles_lues[type].map((r) => (
                        <li key={r.id}>
                          <strong>{r.id}</strong> — <Texte>{r.phrase}</Texte>
                        </li>
                      ))}
                    </ul>
                  </Carte>
                ))}
              </Grille>
              <Legende>
                {tr(
                  "Les {n} autres règles ({l}) calculent des grandeurs ou limitent des courants internes que la simulation ne calcule pas : elles se consultent ci-dessous.",
                  "The {n} other rules ({l}) compute quantities or limit internal currents that the simulation does not compute: they can be browsed below.",
                  { n: o.autres_regles.length, l: o.autres_regles.join(", ") }
                )}
              </Legende>
              <Message type="info">{o.verification}</Message>

              <h3>{tr("Parcourir toutes les règles", "Browse all the rules")}</h3>
              <Onglets
                onglets={[
                  { titre: tr("Règles de l'ontologie ({n})", "Ontology rules ({n})", { n: o.regles.length }), contenu: () => <ParcourirRegles o={o} /> },
                  { titre: tr("Règles de la logique floue ({n})", "Fuzzy-logic rules ({n})", { n: o.regles_floues.length }), contenu: () => <ReglesFloues o={o} /> },
                ]}
              />

              <h2>{tr("D. Comment l'ontologie intervient dans les stratégies ?", "D. How does the ontology come into the strategies?")}</h2>
              <Tableau
                colonnes={[{ cle: "nom", titre: tr("Stratégie", "Strategy") }, { cle: "usage", titre: tr("Usage de l'ontologie", "Use of the ontology") }]}
                lignes={o.usages}
              />
              <p>
                <strong>
                  {tr("Les deux façons d'associer l'ontologie à un réseau de neurones", "The two ways of combining the ontology with a neural network")}
                </strong>
              </p>
              <Grille colonnes={2}>
                {o.chaines.map((chaine, i) => (
                  <Carte key={i}>
                    <Texte>{chaine.titre}</Texte>
                    <Flux etapes={chaine.chaine} />
                    <Legende>{chaine.legende}</Legende>
                  </Carte>
                ))}
              </Grille>
              <Legende>
                {tr(
                  "Pour toutes les stratégies, l'ontologie sert aussi à expliquer : chaque décision est relue avec ses règles dans la page « Pourquoi cette décision ? ».",
                  "For all strategies, the ontology is also used to explain: each decision is re-read with its rules on the “Why this decision?” page."
                )}
              </Legende>

              <TesterLesRegles />
            </>
          )
        }
      </Attente>

      <PiedNavigation page="/connaissances" />
    </>
  );
}
