// Lancer une simulation : le cycle, les stratégies, le lancement. Toutes les
// stratégies choisies sont simulées sur le même cycle avec le même modèle physique
// du HESS ; l'analyse se fait ensuite dans « Résultats de simulation ».

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { envoyer, lire, useCalcul, useDonnees } from "../api.js";
import {
  Attente, Carte, Case, Chargement, Curseur, Depliant, Flux, Grille, Indicateur, Legende, Message, Nombre, PiedNavigation,
  Radios, Selection,
} from "../composants.jsx";
import { COULEURS } from "../couleurs.js";
import { useEtat } from "../etat.jsx";
import { useLangue } from "../langue.jsx";

const CLE_EN_COURS = "2smart.simulation-en-cours";
const DUREE_REFERENCE_MIN = 208;
const NS_MLP = "EMS_MLP_neurosymbolic";
const NS_LSTM = "EMS_LSTM_neurosymbolic";
const PRECISIONS = { rapide: ["Rapide", "Fast"], normale: ["Normale", "Standard"], fine: ["Fine", "Fine"] };
const TEMPS = { rapide: ["rapide", "fast"], moyen: ["moyen", "medium"], long: ["long", "slow"] };

// Compte rendu d'une simulation : attente pendant le calcul, puis lien vers les résultats.
function Suivi({ suivi }) {
  const { tr, nombre } = useLangue();
  if (suivi.etat === "en_cours") {
    return (
      <Carte>
        <Chargement
          texte={tr(
            "Simulation en cours depuis {m} min {s} s… (cela peut prendre plusieurs minutes)",
            "Simulation running for {m} min {s} s… (this may take several minutes)",
            { m: Math.floor((suivi.ecoule_s || 0) / 60), s: Math.floor((suivi.ecoule_s || 0) % 60) }
          )}
        />
        <Legende>
          {tr(
            "Vous pouvez consulter les autres pages pendant le calcul : revenez ici pour voir quand il est terminé.",
            "You can browse the other pages during the calculation: come back here to see when it is finished."
          )}
        </Legende>
      </Carte>
    );
  }
  if (suivi.etat === "echec") {
    return (
      <Message type="erreur">
        {tr("La simulation a échoué : {m}", "The simulation failed: {m}", { m: suivi.message || "" })}
      </Message>
    );
  }
  return (
    <>
      <Carte>
        <h3>{tr("✓ Simulation terminée", "✓ Simulation complete")}</h3>
        <p>
          <strong>{tr("{n} stratégies simulées", "{n} strategies simulated", { n: suivi.nb_strategies })}</strong>{" "}
          {tr("sur « {c} » ({p}) · durée : {m} min {s} s", "on “{c}” ({p}) · time: {m} min {s} s", {
            c: suivi.cycle, p: suivi.pertes_dans_soc ? tr("pertes comptées", "losses counted") : tr("sans pertes", "without losses"),
            m: Math.floor(suivi.duree_s / 60), s: Math.floor(suivi.duree_s % 60),
          })}
        </p>
        <Legende>
          {tr(
            "Les pages d'analyse affichent maintenant cette simulation (« Ma dernière simulation » dans leur choix de cycle).",
            "The analysis pages now show this simulation (“My last simulation” in their cycle selector)."
          )}
        </Legende>
        {(suivi.non_chargees || []).map((nom) => (
          <Message key={nom} type="erreur">
            {tr("{s} n'a pas pu être chargée et n'a pas été simulée.", "{s} could not be loaded and was not simulated.", { s: nom })}
          </Message>
        ))}
        <p>
          <Link className="bouton principal" to="/resultats">
            {tr("Voir les résultats →", "See the results →")}
          </Link>
        </p>
      </Carte>
      {suivi.durees && suivi.durees.length ? (
        <Depliant titre={tr("Durée de calcul par stratégie", "Computing time per strategy")}>
          <ul>
            {suivi.durees.map((d) => (
              <li key={d.nom}>
                {d.nom}
                {tr(" : ", ": ")}
                {nombre(d.duree_s, 0)} s
              </li>
            ))}
          </ul>
        </Depliant>
      ) : null}
    </>
  );
}

function Formulaire({ options }) {
  const { tr, nombre, langue } = useLangue();
  const { prepare, materiel, adopterSimulation } = useEtat();

  const cycles = [
    ...(prepare ? [{ valeur: prepare.id, nom: tr("Le cycle que vous avez préparé", "The cycle you prepared") }] : []),
    ...options.cycles.map((c) => ({ valeur: c.id, nom: c.nom })),
  ];
  const [cycleChoisi, setCycleChoisi] = useState(null);
  const cycle = cycles.some((c) => c.valeur === cycleChoisi) ? cycleChoisi : cycles.length ? cycles[0].valeur : null;

  const [soc, setSoc] = useState({ eb: 100, pb: 100 });
  const [precision, setPrecision] = useState("normale");
  const [pertes, setPertes] = useState({
    activer: false, r_eb_mohm: Math.round(options.pertes.r_eb_mohm * 10) / 10, r_pb_mohm: Math.round(options.pertes.r_pb_mohm * 10) / 10,
    mode: "courbe", rendement_pct: 95.5,
  });
  const toutes = options.familles.flatMap((f) => f.strategies);
  const [cochees, setCochees] = useState(() => toutes.filter((s) => !s.reference).map((s) => s.cle));
  const [suivi, setSuivi] = useState(() => {
    const identifiant = sessionStorage.getItem(CLE_EN_COURS);
    return identifiant ? { id: identifiant, etat: "en_cours" } : null;
  });
  const [erreur, setErreur] = useState(null);

  // NS-MLP reçoit en entrée les prévisions de NS-LSTM.
  const ajoutNsLstm = cochees.includes(NS_MLP) && !cochees.includes(NS_LSTM);
  const choisies = ajoutNsLstm ? [...cochees, NS_LSTM] : cochees;
  const nbTotal = choisies.length + toutes.filter((s) => s.reference).length;

  const resume = useCalcul(cycle ? "/simulation/resume" : null, { cycle, materiel }, 0);
  const diagnostic = useDonnees("/simulation/diagnostic", { soc_eb0: soc.eb / 100, soc_pb0: soc.pb / 100, nb_strategies: nbTotal }, { conserver: true });

  // Durée estimée : poids des stratégies choisies × précision, à proportion de la longueur du
  // cycle (les poids sont ceux du cycle Artemis de référence, environ 208 min).
  const facteur = (options.precisions.find((p) => p.cle === precision) || { facteur: 1 }).facteur;
  const longueur = resume.donnees ? resume.donnees.duree_min / DUREE_REFERENCE_MIN : 1;
  const poids = toutes.filter((s) => choisies.includes(s.cle)).reduce((somme, s) => somme + s.poids, 0) * facteur * longueur;
  const enCours = suivi && suivi.etat === "en_cours";

  // Suivi du calcul : le serveur est interrogé toutes les deux secondes jusqu'à la fin.
  const identifiantSuivi = enCours ? suivi.id : null;
  useEffect(() => {
    if (!identifiantSuivi) return undefined;
    let actif = true;
    const interroger = () =>
      lire(`/simulations/${identifiantSuivi}`, null, langue, { garder: false })
        .then((etat) => {
          if (!actif) return;
          setSuivi(etat);
          if (etat.etat !== "en_cours") sessionStorage.removeItem(CLE_EN_COURS);
          if (etat.etat === "terminee") adopterSimulation(etat.id);
        })
        .catch((e) => {
          if (!actif) return;
          sessionStorage.removeItem(CLE_EN_COURS);
          setSuivi({ id: identifiantSuivi, etat: "echec", message: e.message });
        });
    interroger();
    const minuteur = setInterval(interroger, 2000);
    return () => {
      actif = false;
      clearInterval(minuteur);
    };
  }, [identifiantSuivi, langue, adopterSimulation]);

  const lancer = async () => {
    setErreur(null);
    try {
      const reponse = await envoyer("/simulations", {
        cycle, strategies: choisies, soc_eb0: soc.eb / 100, soc_pb0: soc.pb / 100, precision, materiel,
        pertes: {
          activer: pertes.activer, r_eb_mohm: pertes.r_eb_mohm, r_pb_mohm: pertes.r_pb_mohm,
          rendement_constant: pertes.mode === "constant" ? pertes.rendement_pct / 100 : null,
        },
      }, langue);
      sessionStorage.setItem(CLE_EN_COURS, reponse.id);
      setSuivi(reponse);
    } catch (e) {
      setErreur(e.message);
    }
  };

  const cocher = (cle) => (coche) => setCochees(coche ? [...cochees, cle] : cochees.filter((c) => c !== cle));
  const aideResistance = (r, serie, parallele, decimales) =>
    tr("{r} mΩ par cellule × {s} en série / {p} en parallèle", "{r} mΩ per cell × {s} in series / {p} in parallel", {
      r: nombre(r, decimales), s: serie, p: parallele,
    });

  if (!cycle) {
    return (
      <>
        <Message type="alerte">
          {tr(
            "Aucun cycle disponible. Commencez par la page « Préparer une simulation ».",
            "No cycle available. Start with the “Prepare a simulation” page."
          )}
        </Message>
        <Link className="bouton" to="/preparer">
          {tr("Aller à la préparation", "Go to preparation")}
        </Link>
      </>
    );
  }

  return (
    <>
      <h2>{tr("1 · Le cycle", "1 · The cycle")}</h2>
      <Carte>
        <div className="rangee">
          <Selection titre={tr("Cycle à simuler", "Cycle to simulate")} valeur={cycle} onChange={setCycleChoisi} options={cycles} />
          <Link className="bouton etroit" to="/preparer">
            {tr("Préparer un autre cycle", "Prepare another cycle")}
          </Link>
        </div>
        <Attente etat={resume}>
          {(r) => (
            <>
              <Grille colonnes={4}>
                <Indicateur titre={tr("Durée", "Duration")} valeur={`${nombre(r.duree_min, 0)} min`} />
                <Indicateur titre={tr("Distance", "Distance")} valeur={r.distance_km === null ? "—" : `${nombre(r.distance_km, 0)} km`} />
                <Indicateur titre={tr("Demande maximale", "Peak demand")} valeur={`${nombre(r.demande_max_kw, 1)} kW`} />
                <Indicateur titre={tr("Freinage maximal", "Peak braking")} valeur={`${nombre(r.freinage_max_kw, 1)} kW`} />
              </Grille>
              <Legende>
                {tr(
                  "Énergie nette demandée : {a} kWh, pour {b} kWh utilisables dans le HESS entre 100 % et le SOC minimal.",
                  "Net energy demanded: {a} kWh, for {b} kWh usable in the HESS between 100 % and the minimum SOC.",
                  { a: nombre(r.energie_nette_kwh, 2), b: nombre(r.energie_utile_kwh, 2) }
                )}
              </Legende>
              {r.trop_exigeant ? (
                <Message type="alerte">
                  {tr(
                    "Ce cycle demande plus d'énergie que le HESS n'en contient : toutes les stratégies manqueront d'énergie avant la fin. Réduisez le nombre de répétitions.",
                    "This cycle demands more energy than the HESS contains: every strategy will run out of energy before the end. Reduce the number of repetitions."
                  )}
                </Message>
              ) : null}
            </>
          )}
        </Attente>

        <Depliant titre={tr("États de charge initiaux et précision du calcul", "Initial states of charge and calculation precision")}>
          <Grille colonnes={2}>
            <Curseur titre={tr("SOC initial de la batterie Énergie (%)", "Initial SOC of the Energy battery (%)")} valeur={soc.eb} onChange={(v) => setSoc({ ...soc, eb: v })} min={20} max={100} unite=" %" />
            <Curseur titre={tr("SOC initial de la batterie Puissance (%)", "Initial SOC of the Power battery (%)")} valeur={soc.pb} onChange={(v) => setSoc({ ...soc, pb: v })} min={20} max={100} unite=" %" />
          </Grille>
          <Radios
            titre={tr("Précision du calcul (plus fin = plus long)", "Calculation precision (finer = slower)")} valeur={precision} onChange={setPrecision}
            options={options.precisions.map((p) => ({ valeur: p.cle, nom: tr(...PRECISIONS[p.cle]) }))}
            aide={tr(
              "Pas avec lequel le filtre de sécurité cherche la répartition la plus proche de celle demandée : 0,5 %, 0,2 % ou 0,1 % de la puissance.",
              "Step with which the safety filter searches for the split closest to the requested one: 0.5 %, 0.2 % or 0.1 % of the power."
            )}
          />
          <p>
            <strong>{tr("Hypothèses de la simulation", "Simulation assumptions")}</strong>
          </p>
          <ul>
            {options.hypotheses.map((hypothese, i) => (
              <li key={i}>{hypothese}</li>
            ))}
          </ul>
        </Depliant>

        <Depliant titre={tr("Pertes : non comptées par défaut, comme dans l'article de référence", "Losses: not counted by default, as in the reference paper")}>
          <Case
            titre={tr("Compter les pertes dans le calcul des états de charge", "Count the losses when computing the states of charge")}
            coche={pertes.activer} onChange={(v) => setPertes({ ...pertes, activer: v })}
            aide={tr(
              "Chaque batterie fournit alors aussi ses pertes par effet Joule, et la batterie Énergie celles du convertisseur.",
              "Each battery then also supplies its Joule losses, and the Energy battery those of the converter."
            )}
          />
          <Grille colonnes={3}>
            <Nombre
              titre={tr("Résistance interne de la batterie Énergie (mΩ)", "Internal resistance of the Energy battery (mΩ)")}
              valeur={pertes.r_eb_mohm} onChange={(v) => setPertes({ ...pertes, r_eb_mohm: v })} min={1} max={5000} pas={10} inactif={!pertes.activer}
              aide={aideResistance(options.pertes.r_cellule_eb_mohm, options.pertes.eb_serie, options.pertes.eb_parallele, 0)}
            />
            <Nombre
              titre={tr("Résistance interne de la batterie Puissance (mΩ)", "Internal resistance of the Power battery (mΩ)")}
              valeur={pertes.r_pb_mohm} onChange={(v) => setPertes({ ...pertes, r_pb_mohm: v })} min={1} max={5000} pas={10} inactif={!pertes.activer}
              aide={aideResistance(options.pertes.r_cellule_pb_mohm, options.pertes.pb_serie, options.pertes.pb_parallele, 1)}
            />
            <Radios
              titre={tr("Rendement du convertisseur", "Converter efficiency")} valeur={pertes.mode} onChange={(v) => setPertes({ ...pertes, mode: v })}
              inactif={!pertes.activer}
              options={[
                { valeur: "courbe", nom: tr("Courbe mesurée (article, fig. 33)", "Measured curve (paper, fig. 33)") },
                { valeur: "constant", nom: tr("Valeur constante", "Constant value") },
              ]}
            />
          </Grille>
          {pertes.mode === "constant" ? (
            <Curseur
              titre={tr("Rendement constant (%)", "Constant efficiency (%)")} valeur={pertes.rendement_pct}
              onChange={(v) => setPertes({ ...pertes, rendement_pct: v })} min={80} max={100} pas={0.1} unite=" %" affichage={(v) => nombre(v, 1)}
            />
          ) : null}
          <Legende>
            {tr(
              "Courbe mesurée : 91,5 % à 1,2 kW, 95,5 % vers 2,5 kW. Ici le convertisseur ne traite qu'environ 10 % de la puissance de la batterie Énergie, souvent moins de 1,2 kW : la valeur à 1,2 kW est alors conservée, ce qui sous-estime légèrement ses pertes.",
              "Measured curve: 91.5 % at 1.2 kW, 95.5 % around 2.5 kW. Here the converter only processes about 10 % of the Energy battery's power, often less than 1.2 kW: the value at 1.2 kW is then kept, which slightly underestimates its losses."
            )}
          </Legende>
        </Depliant>
      </Carte>

      <h2>{tr("2 · Les stratégies", "2 · The strategies")}</h2>
      <Legende>
        {tr(
          "Le modèle physique et la logique floue sont toujours simulés : ils servent de référence. Cochez les stratégies à apprentissage à ajouter.",
          "The physical model and fuzzy logic are always simulated: they serve as references. Tick the learning-based strategies to add."
        )}
      </Legende>
      <Grille colonnes={options.familles.length}>
        {options.familles.map((famille) => (
          <Carte key={famille.nom}>
            <h4>{famille.nom}</h4>
            {famille.strategies.map((s) =>
              s.reference ? (
                <Case key={s.cle} titre={s.nom} coche inactif aide={tr("Toujours simulée : stratégie de référence", "Always simulated: reference strategy")} />
              ) : (
                <Case
                  key={s.cle} titre={s.nom} coche={cochees.includes(s.cle)} onChange={cocher(s.cle)}
                  aide={tr("Temps de calcul : {t}", "Computing time: {t}", { t: tr(...(TEMPS[s.temps] || ["", ""])) })}
                />
              )
            )}
          </Carte>
        ))}
      </Grille>
      {ajoutNsLstm ? (
        <Legende>{tr("NS-LSTM sera aussi simulé : NS-MLP utilise ses prévisions.", "NS-LSTM will also be simulated: NS-MLP uses its forecasts.")}</Legende>
      ) : null}

      <h2>{tr("3 · Lancer", "3 · Run")}</h2>
      <Carte>
        <p>
          <strong>{tr("Toutes les stratégies sont comparées dans les mêmes conditions", "All strategies are compared under the same conditions")}</strong>
        </p>
        <Flux
          etapes={[
            { texte: tr("Même profil de conduite", "Same driving profile"), couleur: COULEURS.reference },
            { texte: tr("{n} stratégies EMS", "{n} EMS strategies", { n: nbTotal }), couleur: COULEURS.decision },
            { texte: tr("Même modèle physique du HESS", "Same physical model of the HESS"), couleur: COULEURS.secondaire },
            { texte: tr("SOC · puissances · courants · pertes", "SOC · powers · currents · losses"), couleur: COULEURS.eb },
          ]}
        />
        <Legende>
          {tr(
            "À chaque instant, chaque stratégie propose sa répartition de la puissance ; le même filtre de sécurité et le même modèle de batteries calculent ensuite l'évolution du système. Seule la décision change d'une stratégie à l'autre.",
            "At each time step, each strategy proposes its power split; the same safety filter and the same battery model then compute how the system evolves. Only the decision changes from one strategy to another."
          )}
        </Legende>

        {diagnostic.donnees ? (
          <>
            {diagnostic.donnees.alertes.map((alerte, i) => (
              <Message key={i} type="alerte">{alerte}</Message>
            ))}
            <Depliant titre={tr("Vérifications avant lancement (base de connaissances OntoHESS)", "Checks before running (OntoHESS knowledge base)")}>
              <Grille colonnes={2}>
                {[
                  [tr("Système reconnu", "System recognised"), diagnostic.donnees.contexte],
                  [tr("Contraintes prises en compte", "Constraints taken into account"), diagnostic.donnees.contraintes],
                ].map(([titre, elements]) => (
                  <div key={titre}>
                    <strong>{titre}</strong>
                    <ul>
                      {elements.map((e) => (
                        <li key={e.concept}>
                          {e.reconnu ? "✔️" : "—"} {e.libelle}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </Grille>
              {diagnostic.donnees.conseils.map((conseil, i) => (
                <Message key={i} type="info">{conseil}</Message>
              ))}
              <Message type="succes">{diagnostic.donnees.conclusion}</Message>
            </Depliant>
          </>
        ) : null}

        <Legende>
          {tr("{n} stratégies · précision « {p} » · durée estimée : ", "{n} strategies · “{p}” precision · estimated time: ", {
            n: nbTotal, p: tr(...PRECISIONS[precision]).toLowerCase(),
          }) +
            (poids * 4 >= 1.5
              ? tr("{a} à {b} min, selon l'ordinateur.", "{a} to {b} min, depending on the computer.", {
                  a: Math.max(1, Math.round(poids * 1.5)), b: Math.round(poids * 4),
                })
              : poids
                ? tr("moins d'une minute.", "less than a minute.")
                : tr("quelques secondes.", "a few seconds."))}
        </Legende>
        <Message type="erreur">{erreur}</Message>
        <p>
          <button className="bouton principal" onClick={lancer} disabled={enCours}>
            {tr("🚀 Lancer la simulation", "🚀 Run the simulation")}
          </button>
        </p>
      </Carte>

      {suivi ? <Suivi suivi={suivi} /> : null}
    </>
  );
}

export default function Lancer() {
  const { tr } = useLangue();
  const options = useDonnees("/simulation/options");

  return (
    <>
      <h1>{tr("▶️ Lancer une simulation", "▶️ Run a simulation")}</h1>
      <Legende>
        {tr(
          "Choisir le cycle et les stratégies, puis lancer la même simulation physique pour toutes les stratégies choisies.",
          "Choose the cycle and the strategies, then run the same physical simulation for all the selected strategies."
        )}
      </Legende>
      <Attente etat={options}>{(o) => <Formulaire options={o} />}</Attente>
      <PiedNavigation page="/lancer" />
    </>
  );
}
