// Résultats de simulation : évolution du HESS au cours du cycle pour la stratégie
// choisie — répartition de la puissance, états de charge, charge et décharge de
// chaque composant, pertes.

import { useDonnees } from "../api.js";
import {
  Attente, Depliant, Grille, Indicateur, Legende, Message, PiedNavigation, SelecteurCycle, SelecteurStrategie,
  Tableau, Texte, telechargerCsv, useCycle, useStrategie,
} from "../composants.jsx";
import { useEtat } from "../etat.jsx";
import { GraphiqueChargeDecharge, GraphiquePertes, GraphiquePuissances, GraphiqueSoc } from "../graphiques.jsx";
import { useLangue } from "../langue.jsx";

export default function Resultats() {
  const { tr, nombre, langue } = useLangue();
  const { cycle } = useEtat();
  const infos = useCycle();
  const [strategie, setStrategie] = useStrategie(infos.donnees);
  const base = strategie ? `/cycles/${cycle}/strategies/${strategie}` : null;
  const indicateurs = useDonnees(base && `${base}/indicateurs`);
  const courbes = useDonnees(base && `${base}/courbes`);

  const nomEb = tr("Batterie Énergie", "Energy battery");
  const nomPb = tr("Batterie Puissance", "Power battery");
  const nomConv = tr("Convertisseur", "Converter");
  const noms = { eb: nomEb, pb: nomPb, conv: nomConv };
  const aideReglette = tr(
    "Faites glisser la réglette sous le graphique pour parcourir le cycle.",
    "Drag the slider under the chart to move through the cycle."
  );
  const aideSurvol = tr(
    "Passez le curseur sur une courbe : la bulle donne les valeurs et explique ce qui se passe à cet instant.",
    "Move the cursor over a curve: the tooltip gives the values and explains what is happening at that time."
  );

  const telecharger = (c, nom) => {
    const colonnes = ["temps_min", "P_demande_kW", "P_EB_kW", "P_PB_kW", "P_convertisseur_kW", "SOC_EB_pct", "SOC_PB_pct"];
    const lignes = c.temps_min.map((t, i) => [t, c.demande_kw[i], c.eb_kw[i], c.pb_kw[i], c.conv_kw[i], c.soc_eb_pct[i], c.soc_pb_pct[i]]);
    telechargerCsv(`courbes_${nom}.csv`, colonnes, lignes, langue);
  };

  return (
    <>
      <h1>{tr("📈 Résultats de simulation", "📈 Simulation results")}</h1>
      <Legende>{tr("Évolution du HESS au cours du cycle pour la stratégie choisie.", "How the HESS evolves over the cycle for the selected strategy.")}</Legende>
      <div className="rangee">
        <SelecteurStrategie infos={infos.donnees} valeur={strategie} onChange={setStrategie} />
        <SelecteurCycle />
      </div>

      <Attente etat={indicateurs}>
        {(d) => {
          const m = d.metriques;
          return (
            <>
              <Message type="alerte">{d.avertissement}</Message>
              <Grille colonnes={3}>
                <Indicateur
                  titre={tr("M1 · Énergie consommée", "M1 · Energy consumed")} valeur={`${nombre(m.energie_km_wh, 1)} Wh/km`}
                  aide={tr("{e} kWh sur le cycle, pertes comprises", "{e} kWh over the cycle, losses included", { e: nombre(m.energie_consommee_wh / 1000, 2) })}
                />
                <Indicateur
                  titre={tr("M2 · Rendement du HESS", "M2 · HESS efficiency")} valeur={`${nombre(m.rendement_hess * 100, 2)} %`}
                  aide={tr("Énergie de traction fournie, divisée par cette énergie augmentée des pertes", "Traction energy delivered, divided by that energy plus the losses")}
                />
                <Indicateur
                  titre={tr("M3 · Écart entre les SOC", "M3 · Gap between SOCs")} valeur={`${nombre(m.rmse_delta_soc * 100, 1)} pts`}
                  aide={tr("Écart quadratique moyen ; écart maximal {x} points", "Root-mean-square gap; maximum gap {x} points", { x: nombre(m.delta_soc_max * 100, 1) })}
                />
                <Indicateur titre={tr("M4 · Pertes du convertisseur", "M4 · Converter losses")} valeur={`${nombre(m.pertes_convertisseur_wh, 0)} Wh`} />
                <Indicateur
                  titre={tr("M5 · Dépassements de limite", "M5 · Limit violations")} valeur={nombre(m.nb_violations + m.nb_violations_courant, 0)}
                  aide={tr("{a} de SOC, {b} de courant", "{a} of SOC, {b} of current", { a: m.nb_violations, b: m.nb_violations_courant })}
                />
                <Indicateur
                  titre={tr("M6 · Écart de puissance", "M6 · Power shortfall")} valeur={`${nombre(m.rmse_puissance_kw, 2)} kW`}
                  aide={tr(
                    "Écart quadratique moyen entre puissance fournie et demandée en traction ; écart maximal {x} kW",
                    "Root-mean-square gap between delivered and demanded traction power; maximum gap {x} kW",
                    { x: nombre(m.ecart_puissance_max_kw, 1) }
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
            <h2>{tr("Comment la demande de puissance est-elle répartie entre les deux batteries ?", "How is the power demand shared between the two batteries?")}</h2>
            <Legende>
              {aideSurvol} {aideReglette}
            </Legende>
            <GraphiquePuissances courbes={c} limites={infos.donnees.limites} />

            <h2>{tr("Évolution des états de charge", "State of charge over time")}</h2>
            <Legende>{aideSurvol}</Legende>
            <GraphiqueSoc courbes={c} limites={infos.donnees.limites} avecMaximum />

            <h2>{tr("Charge et décharge des batteries et du convertisseur", "Charge and discharge of the batteries and the converter")}</h2>
            <Legende>
              {tr(
                "Au-dessus de zéro, le composant se décharge : il fournit de la puissance. En dessous, il se recharge : il reçoit l'énergie récupérée au freinage. Les pointillés marquent ses limites.",
                "Above zero the component discharges: it supplies power. Below zero it charges: it receives the energy recovered while braking. The dotted lines mark its limits."
              )}{" "}
              {aideSurvol} {aideReglette}
            </Legende>
            <GraphiqueChargeDecharge courbes={c} limites={infos.donnees.limites} />
          </>
        )}
      </Attente>

      {indicateurs.donnees ? (
        <>
          <Tableau
            colonnes={[
              { cle: "nom", titre: tr("Composant", "Component") },
              { cle: "decharge", titre: tr("Énergie fournie en décharge (Wh)", "Energy supplied while discharging (Wh)"), droite: true },
              { cle: "recharge", titre: tr("Énergie reçue en recharge (Wh)", "Energy received while charging (Wh)"), droite: true },
              { cle: "t_decharge", titre: tr("Temps en décharge", "Time discharging"), droite: true },
              { cle: "t_recharge", titre: tr("Temps en recharge", "Time charging"), droite: true },
              { cle: "pic", titre: tr("Puissance maximale atteinte (kW)", "Peak power reached (kW)"), droite: true },
              { cle: "limites", titre: tr("Limites décharge / recharge (kW)", "Discharge / charge limits (kW)"), droite: true },
            ]}
            lignes={indicateurs.donnees.composants.map((c) => ({
              nom: noms[c.cle],
              decharge: nombre(c.energie_decharge_wh, 0),
              recharge: nombre(c.energie_recharge_wh, 0),
              t_decharge: `${nombre(c.temps_decharge * 100, 0)} %`,
              t_recharge: `${nombre(c.temps_recharge * 100, 0)} %`,
              pic: `${nombre(c.p_max_kw, 2)} / ${nombre(c.p_min_kw, 2)}`,
              limites: `${nombre(c.limite_decharge_kw, 2)} / ${nombre(c.limite_recharge_kw, 2)}`,
            }))}
          />
          <Legende>
            {tr(
              "Le convertisseur, placé en série avec la batterie Énergie, ne traite que la différence de tension entre les deux batteries : sa puissance vaut (V_EB − V_PB)·I_EB, environ {p} % de celle de la batterie Énergie, dont elle suit le sens.",
              "The converter, in series with the Energy battery, only processes the voltage difference between the two batteries: its power is (V_EB − V_PB)·I_EB, about {p} % of the Energy battery's power, and has the same sign.",
              { p: nombre(indicateurs.donnees.part_convertisseur * 100, 1) }
            )}
          </Legende>
        </>
      ) : null}

      <Attente etat={courbes}>
        {(c) => (
          <>
            <h2>{tr("Pertes", "Losses")}</h2>
            <Legende>
              {tr(
                "Pertes cumulées au fil du cycle : échauffement de chaque batterie (R·I²) et pertes du convertisseur. Au survol : les pertes cumulées, et la puissance perdue à cet instant.",
                "Cumulative losses over the cycle: heating of each battery (R·I²) and converter losses. On hover: the cumulative losses, and the power being lost at that time."
              )}
            </Legende>
            <GraphiquePertes courbes={c} />
          </>
        )}
      </Attente>

      {indicateurs.donnees ? (
        <>
          <Grille colonnes={4}>
            <Indicateur titre={nomEb} valeur={`${nombre(indicateurs.donnees.pertes.eb_wh, 0)} Wh`} />
            <Indicateur titre={nomPb} valeur={`${nombre(indicateurs.donnees.pertes.pb_wh, 0)} Wh`} />
            <Indicateur titre={nomConv} valeur={`${nombre(indicateurs.donnees.pertes.conv_wh, 0)} Wh`} />
            <Indicateur titre={tr("Pertes totales", "Total losses")} valeur={`${nombre(indicateurs.donnees.pertes.total_wh, 0)} Wh`} />
          </Grille>
          <Message type={indicateurs.donnees.pertes.bilan.type}>{indicateurs.donnees.pertes.bilan.texte}</Message>
          <Depliant titre={tr("Hypothèses du calcul des pertes", "Assumptions of the loss calculation")}>
            <Texte>
              {tr(
                "- Résistance interne de chaque pack, calculée à partir des cellules : {reb} mΩ × {seb}/{peb} pour la batterie Énergie, {rpb} mΩ × {spb}/{ppb} pour la batterie Puissance.\n- Convertisseur : rendement mesuré dans l'article de référence (fig. 33), de 91,5 % à 1,2 kW à 95,5 % vers 2,5 kW, appliqué à la seule puissance qu'il traite ; en dessous de 1,2 kW, la valeur à 1,2 kW est conservée.\n- Tensions constantes, sans variation avec le SOC.",
                "- Internal resistance of each pack, computed from the cells: {reb} mΩ × {seb}/{peb} for the Energy battery, {rpb} mΩ × {spb}/{ppb} for the Power battery.\n- Converter: efficiency measured in the reference paper (fig. 33), from 91.5 % at 1.2 kW to 95.5 % around 2.5 kW, applied only to the power it processes; below 1.2 kW, the value at 1.2 kW is kept.\n- Constant voltages, with no variation with SOC.",
                {
                  reb: nombre(indicateurs.donnees.hypotheses_pertes.r_eb_mohm, 0), seb: indicateurs.donnees.hypotheses_pertes.eb_serie,
                  peb: indicateurs.donnees.hypotheses_pertes.eb_parallele, rpb: nombre(indicateurs.donnees.hypotheses_pertes.r_pb_mohm, 1),
                  spb: indicateurs.donnees.hypotheses_pertes.pb_serie, ppb: indicateurs.donnees.hypotheses_pertes.pb_parallele,
                }
              )}
            </Texte>
          </Depliant>
          {courbes.donnees ? (
            <button className="bouton" onClick={() => telecharger(courbes.donnees, strategie)}>
              {tr("Télécharger les courbes de {s} (CSV)", "Download the curves of {s} (CSV)", { s: indicateurs.donnees.strategie.nom })}
            </button>
          ) : null}
        </>
      ) : null}

      <PiedNavigation page="/resultats" />
    </>
  );
}
