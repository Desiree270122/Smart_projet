// Tableau de bord : comprendre en quelques secondes ce qui se passe dans le HESS
// pour une stratégie — état du système, répartition de la puissance, synthèse rédigée.

import { Link } from "react-router-dom";
import { useDonnees } from "../api.js";
import {
  Attente, Carte, Depliant, Flux, Formule, Grille, Indicateur, Legende, PiedNavigation, SelecteurCycle,
  SelecteurStrategie, Texte, useCycle, useStrategie,
} from "../composants.jsx";
import { COULEURS } from "../couleurs.js";
import { useEtat } from "../etat.jsx";
import { GraphiquePuissances, GraphiqueSoc } from "../graphiques.jsx";
import { useLangue } from "../langue.jsx";

export default function TableauDeBord() {
  const { tr, nombre } = useLangue();
  const { cycle } = useEtat();
  const infos = useCycle();
  const [strategie, setStrategie] = useStrategie(infos.donnees);
  const base = strategie ? `/cycles/${cycle}/strategies/${strategie}` : null;
  const indicateurs = useDonnees(base && `${base}/indicateurs`);
  const courbes = useDonnees(base && `${base}/courbes`);
  const aideSurvol = tr(
    "Passez le curseur sur une courbe : la bulle donne les valeurs et explique ce qui se passe à cet instant.",
    "Move the cursor over a curve: the tooltip gives the values and explains what is happening at that time."
  );

  return (
    <>
      <h1>{tr("2SMART · Gestion d'énergie d'un HESS", "2SMART · Energy management of a HESS")}</h1>
      <p className="accroche">
        {tr(
          "Deux batteries complémentaires, une décision à chaque seconde : quelle part de la puissance confier à chacune ? L'application simule, compare et explique cette décision pour sept stratégies de gestion d'énergie (EMS).",
          "Two complementary batteries, one decision every second: what share of the power should each one supply? The application simulates, compares and explains this decision for seven energy management strategies (EMS)."
        )}
      </p>
      <Flux
        etapes={[
          { texte: tr("Demande du véhicule", "Vehicle demand"), couleur: COULEURS.demande },
          { texte: tr("EMS : décision alpha", "EMS: alpha decision"), couleur: COULEURS.decision },
          { texte: tr("Batterie Énergie (1 − alpha)", "Energy battery (1 − alpha)"), couleur: COULEURS.eb },
          { texte: tr("Batterie Puissance (alpha)", "Power battery (alpha)"), couleur: COULEURS.pb },
          { texte: tr("Convertisseur en série (EB)", "Series converter (EB)"), couleur: COULEURS.convertisseur },
          { texte: tr("Bus DC et moteur", "DC bus and motor"), couleur: COULEURS.secondaire },
        ]}
      />

      <div className="rangee">
        <SelecteurStrategie infos={infos.donnees} valeur={strategie} onChange={setStrategie} />
        <SelecteurCycle />
        <Link className="bouton etroit" to="/preparer">
          {tr("▶ Autre cycle", "▶ Other cycle")}
        </Link>
      </div>

      <Attente etat={indicateurs}>
        {(d) => {
          const m = d.metriques;
          const depassements = m.nb_violations + m.nb_violations_courant;
          return (
            <>
              <h2>{tr("État du système avec {s}", "System state with {s}", { s: d.strategie.nom })}</h2>
              <Grille colonnes={3}>
                <Indicateur
                  titre={tr("Puissance demandée (max)", "Power demand (max)")} valeur={`${nombre(d.demande_max_kw, 1)} kW`}
                  aide={tr("Moyenne en traction : {p} kW", "Mean in traction: {p} kW", { p: nombre(d.demande_moyenne_traction_kw, 1) })}
                />
                <Indicateur titre={tr("SOC final · batterie Énergie", "Final SOC · Energy battery")} valeur={`${nombre(d.soc_eb_final * 100, 1)} %`} />
                <Indicateur titre={tr("SOC final · batterie Puissance", "Final SOC · Power battery")} valeur={`${nombre(d.soc_pb_final * 100, 1)} %`} />
                <Indicateur titre={tr("Rendement du HESS (estimé)", "HESS efficiency (estimated)")} valeur={`${nombre(m.rendement_hess * 100, 2)} %`} />
                <Indicateur
                  titre={tr("Écart moyen entre les SOC", "Mean gap between the SOCs")} valeur={`${nombre(m.rmse_delta_soc * 100, 1)} pts`}
                  aide={tr("Écart quadratique moyen ; écart maximal : {v} points", "Root-mean-square gap; maximum gap: {v} points", { v: nombre(m.delta_soc_max * 100, 1) })}
                />
                <Indicateur
                  titre={tr("Contraintes et demande", "Constraints and demand")}
                  valeur={d.contraintes_respectees ? tr("✓ respectées", "✓ met") : tr("✗ non respectées", "✗ not met")}
                  aide={tr(
                    "{v} dépassement(s) de SOC ou de courant ; {e} Wh de demande non fournie",
                    "{v} SOC or current limit violation(s); {e} Wh of demand not supplied",
                    { v: depassements, e: nombre(m.energie_non_servie_wh, 0) }
                  )}
                />
              </Grille>
            </>
          );
        }}
      </Attente>

      <Attente etat={courbes}>
        {(c) => (
          <>
            <h2>{tr("Évolution des états de charge", "State of charge over time")}</h2>
            <GraphiqueSoc courbes={c} limites={infos.donnees.limites} />
            <h2>{tr("Répartition de la puissance", "Power split")}</h2>
            <Legende>
              {tr(
                "Le véhicule demande une puissance ; l'EMS la répartit entre les deux batteries. Faites glisser la réglette sous le graphique pour parcourir le cycle.",
                "The vehicle demands a power; the EMS splits it between the two batteries. Drag the slider below the chart to move through the cycle."
              )}{" "}
              {aideSurvol}
            </Legende>
            <GraphiquePuissances courbes={c} limites={infos.donnees.limites} />
          </>
        )}
      </Attente>

      {indicateurs.donnees ? (
        <>
          <h2>{tr("Analyse de la simulation", "Simulation summary")}</h2>
          <Carte>
            <Texte>{indicateurs.donnees.synthese.join(" ")}</Texte>
          </Carte>
        </>
      ) : null}

      <Grille colonnes={3}>
        <Link className="bouton principal" to="/explication">
          {tr("🔍 Pourquoi ces décisions ?", "🔍 Why these decisions?")}
        </Link>
        <Link className="bouton" to="/comparaison">
          {tr("⚖️ Comparer les stratégies", "⚖️ Compare the strategies")}
        </Link>
        <Link className="bouton" to="/preparer">
          {tr("📈 Simuler un autre cycle", "📈 Simulate another cycle")}
        </Link>
      </Grille>

      <Depliant titre={tr("Le projet en bref", "The project in brief")}>
        <Texte>
          {tr(
            "- **Le problème** : un véhicule électrique équipé de deux batteries complémentaires doit décider, à chaque instant, laquelle fournit la puissance demandée.\n- **Pourquoi c'est difficile** : les objectifs se contredisent (autonomie, durée de vie, rendement), les contraintes physiques sont strictes et les cycles de conduite très variables.\n- **La réponse de 2SMART** : comparer quatre familles d'approches (règles fixes, ontologie seule, apprentissage seul, hybride neuro-symbolique) et expliquer chaque décision, sous le contrôle d'un filtre physique de sécurité.\n- **L'architecture** : cascade à source de courant contrôlée (Fonseca de Freitas et al., IEEE Access 2024) ; le convertisseur, en série, ne traite qu'environ 10 % de la puissance de la batterie Énergie.",
            "- **The problem**: an electric vehicle with two complementary batteries must decide, at every instant, which one supplies the demanded power.\n- **Why it is hard**: the objectives conflict (range, lifetime, efficiency), the physical constraints are strict and driving cycles vary widely.\n- **The 2SMART approach**: compare four families of approaches (fixed rules, ontology only, learning only, neuro-symbolic hybrid) and explain every decision, under the control of a physical safety filter.\n- **The architecture**: controlled current source cascade (Fonseca de Freitas et al., IEEE Access 2024); the converter, in series, processes only about 10 % of the Energy battery's power."
          )}
        </Texte>
        <Grille colonnes={2}>
          <Formule tex={"P_{PB} = \\alpha \\times P_{dem}"} />
          <Formule tex={"P_{EB} = (1 - \\alpha) \\times P_{dem}"} />
        </Grille>
        <Legende>
          {tr(
            "alpha = 0 : toute la puissance vient de la batterie Énergie ; alpha = 1 : toute la puissance vient de la batterie Puissance. La décision passe ensuite par le filtre physique de sécurité.",
            "alpha = 0: all the power comes from the Energy battery; alpha = 1: all the power comes from the Power battery. The decision then goes through the physical safety filter."
          )}
        </Legende>
      </Depliant>

      <PiedNavigation page="/" />
    </>
  );
}
