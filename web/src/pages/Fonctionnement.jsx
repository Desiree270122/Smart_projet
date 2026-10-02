// Fonctionnement des stratégies EMS : comment chacune transforme l'état du HESS
// en une répartition de la puissance, avec quelles connaissances, et ce qu'elle a appris.

import { useDonnees } from "../api.js";
import { Attente, Carte, Depliant, Flux, Formule, Grille, Legende, Message, PiedNavigation, Tableau, Texte } from "../composants.jsx";
import { useLangue } from "../langue.jsx";

function Fiche({ fiche }) {
  const { tr } = useLangue();
  return (
    <Carte>
      <h4>{fiche.nom}</h4>
      <Legende>
        {fiche.nature} · {tr("famille", "family")} « {fiche.famille} »
      </Legende>
      <Flux etapes={fiche.flux} />
      <Depliant titre={tr("Détails", "Details")}>
        <p>
          <strong>{tr("À quoi elle sert", "What it is for")}</strong> — <Texte>{fiche.role}</Texte>
        </p>
        <strong>{tr("Comment elle fonctionne", "How it works")}</strong>
        <ul>
          {fiche.fonctionnement.map((point, i) => (
            <li key={i}>
              <Texte>{point}</Texte>
            </li>
          ))}
        </ul>
        <p>
          <strong>{tr("Ce qu'elle a appris à reproduire", "What it learned to reproduce")}</strong> — <Texte>{fiche.appris}</Texte>
        </p>
        <p>
          <strong>{tr("Grandeurs utilisées", "Quantities used")}</strong> — <Texte>{fiche.grandeurs}</Texte>
        </p>
      </Depliant>
    </Carte>
  );
}

export default function Fonctionnement() {
  const { tr } = useLangue();
  const strategies = useDonnees("/strategies");

  return (
    <>
      <h1>{tr("🧠 Fonctionnement des stratégies EMS", "🧠 How the EMS strategies work")}</h1>
      <Legende>
        {tr(
          "Comment chaque stratégie transforme l'état du HESS en une répartition de la puissance.",
          "How each strategy turns the state of the HESS into a power split."
        )}
      </Legende>

      <Attente etat={strategies}>
        {(d) => {
          const classiques = d.fiches.filter((f) => !f.neuro_symbolique);
          const parCle = Object.fromEntries(d.fiches.map((f) => [f.cle, f]));
          return (
            <>
              <h2>{tr("Le principe commun", "The common principle")}</h2>
              <Flux etapes={d.principe} />
              <Grille colonnes={2}>
                <Formule tex={"P_{PB} = \\alpha \\times P_{dem}"} />
                <Formule tex={"P_{EB} = (1 - \\alpha) \\times P_{dem}"} />
              </Grille>
              <Legende>
                {tr(
                  "Toutes les stratégies donnent la même grandeur, alpha : la part de la puissance confiée à la batterie Puissance. Un filtre de sécurité vérifie ensuite les limites de courant, de puissance et de SOC, et corrige alpha si nécessaire.",
                  "All strategies output the same quantity, alpha: the share of power assigned to the Power battery. A safety filter then checks the current, power and SOC limits, and corrects alpha if necessary."
                )}
              </Legende>
              <Depliant titre={tr("Architecture électrique du HESS (convertisseur à puissance partielle)", "Electrical architecture of the HESS (partial-power converter)")}>
                <Texte>{d.architecture}</Texte>
                <Formule tex={"P_{conv} = (V_{EB} - V_{PB})\\,I_{EB} = P_{EB}\\,\\frac{V_{EB} - V_{PB}}{V_{EB}} \\qquad P_{dem} = P_{EB} + P_{PB}"} />
                <Legende>{d.reference}</Legende>
              </Depliant>

              <h2>{tr("Les sept stratégies", "The seven strategies")}</h2>
              <Grille colonnes={2}>
                {classiques.map((fiche) => (
                  <Fiche key={fiche.cle} fiche={fiche} />
                ))}
              </Grille>

              <h2>{tr("Pourquoi deux stratégies neuro-symboliques ?", "Why two neuro-symbolic strategies?")}</h2>
              <p>
                {tr(
                  "Les deux combinent connaissances expertes et apprentissage, mais pas au même endroit de la chaîne de décision. Les comparer montre ce que change la place donnée aux connaissances expertes.",
                  "Both combine expert knowledge and learning, but not at the same place in the decision chain. Comparing them shows what the place given to expert knowledge changes."
                )}
              </p>
              <Grille colonnes={2}>
                <div>
                  <Fiche fiche={parCle.EMS_MLP_neurosymbolic} />
                  <Message type="info">
                    {tr(
                      "**NS-MLP** — les connaissances expertes sont le **socle de la décision** : les règles floues proposent une répartition, le réseau n'y ajoute qu'une correction limitée, et un garde-fou de l'ontologie (règles R14/R16) protège la réserve de la batterie Puissance.",
                      "**NS-MLP** — expert knowledge is the **basis of the decision**: the fuzzy rules propose a split, the network only adds a limited correction, and an ontology safeguard (rules R14/R16) protects the Power battery's reserve."
                    )}
                  </Message>
                </div>
                <div>
                  <Fiche fiche={parCle.EMS_LSTM_neurosymbolic} />
                  <Message type="info">
                    {tr(
                      "**NS-LSTM** — le réseau exploite **le passé récent du cycle** et reçoit des états déduits par l'ontologie comme informations supplémentaires. Les connaissances expertes éclairent la décision sans la structurer.",
                      "**NS-LSTM** — the network uses **the recent past of the cycle** and receives states inferred by the ontology as additional information. Expert knowledge informs the decision without structuring it."
                    )}
                  </Message>
                </div>
              </Grille>

              <hr className="separateur" />
              <h2>{tr("Les stratégies en un tableau", "The strategies in one table")}</h2>
              <Tableau
                colonnes={[
                  { cle: "nom", titre: tr("Stratégie", "Strategy") },
                  { cle: "famille", titre: tr("Famille", "Family") },
                  { cle: "apprentissage", titre: tr("Apprentissage", "Learning") },
                  { cle: "appris", titre: tr("Ce qu'elle a appris à reproduire", "What it learned to reproduce") },
                  { cle: "memoire", titre: tr("Mémoire du passé", "Memory of the past") },
                  { cle: "ontologie", titre: tr("Usage de l'ontologie", "Use of the ontology") },
                  { cle: "lecture", titre: tr("Comment la décision se lit", "How the decision can be read") },
                ]}
                lignes={d.fiches.map((f) => ({ nom: f.nom, famille: f.famille, ...f.synthese }))}
              />
              <Legende>
                {tr(
                  "Les stratégies à apprentissage ont été réglées à l'avance, sur la première moitié du cycle Artemis ; l'application rejoue leurs décisions. Le détail de l'ontologie et de ses règles est dans « Base de connaissances ».",
                  "The learning-based strategies were tuned beforehand, on the first half of the Artemis cycle; the app replays their decisions. The details of the ontology and its rules are in “Knowledge base”."
                )}
              </Legende>
            </>
          );
        }}
      </Attente>

      <PiedNavigation page="/fonctionnement" />
    </>
  );
}
