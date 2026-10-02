// Préparer une simulation : importer un cycle de conduite, choisir ses colonnes,
// calculer la puissance demandée si besoin, et régler les batteries et le
// convertisseur. Le cycle préparé est ensuite simulé depuis « Lancer une simulation ».

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { envoyer, envoyerFichier, lire, useCalcul, useDonnees } from "../api.js";
import {
  Attente, Case, Curseur, Depliant, Formule, Graphique, Grille, Indicateur, Legende, Message, Nombre, PiedNavigation,
  Radios, Selection, Tableau, axe,
} from "../composants.jsx";
import { COULEURS } from "../couleurs.js";
import { useEtat } from "../etat.jsx";
import { useLangue } from "../langue.jsx";

const AUCUNE = "";

// Champs d'une batterie : [clé, titre (fr, en), pas]
const CHAMPS_CELLULE = [
  ["n_serie", ["Cellules en série", "Cells in series"], 1],
  ["n_parallele", ["Cellules en parallèle", "Cells in parallel"], 1],
  ["masse_cellule", ["Masse d'une cellule (kg)", "Mass of one cell (kg)"], 0.001],
  ["v_cellule", ["Tension d'une cellule (V)", "Voltage of one cell (V)"], 0.1],
  ["i_decharge", ["Courant de décharge d'une cellule (A)", "Discharge current of one cell (A)"], 0.1],
  ["i_recharge", ["Courant de recharge d'une cellule (A)", "Charge current of one cell (A)"], 0.1],
  ["de", ["Densité d'énergie DE (Wh/kg)", "Energy density DE (Wh/kg)"], 1],
  ["capacite", ["Capacité d'une cellule (Ah)", "Capacity of one cell (Ah)"], 0.1],
  ["rint", ["Résistance interne d'une cellule (Ω)", "Internal resistance of one cell (Ω)"], 0.001],
];

// Batteries et convertisseur : caractéristiques des packs calculées à partir des cellules.
function Materiel({ cellules, setCellules }) {
  const { tr, nombre } = useLangue();
  const calcul = useCalcul("/materiel", cellules);
  const regler = (pack, cle) => (valeur) => setCellules({ ...cellules, [pack]: { ...cellules[pack], [cle]: valeur } });
  const noms = { eb: tr("Énergie (EB)", "Energy (EB)"), pb: tr("Puissance (PB)", "Power (PB)"), total: tr("Les deux batteries", "Both batteries") };

  return (
    <>
      <h2>{tr("7. Architecture des batteries", "7. Battery architecture")}</h2>
      <p>
        {tr(
          "Caractéristiques d'une cellule et nombre de cellules en série et en parallèle, pour chacune des deux batteries. Les valeurs par défaut sont celles du projet ; elles sont modifiables.",
          "Characteristics of one cell and number of cells in series and in parallel, for each of the two batteries. The default values are those of the project; they can be changed."
        )}
      </p>
      <Grille colonnes={2}>
        {[
          ["eb", tr("Batterie Énergie (EB)", "Energy battery (EB)")],
          ["pb", tr("Batterie Puissance (PB)", "Power battery (PB)")],
        ].map(([pack, titre]) => (
          <div key={pack}>
            <h4>{titre}</h4>
            {CHAMPS_CELLULE.map(([cle, libelle, pas]) => (
              <Nombre key={cle} titre={tr(...libelle)} valeur={cellules[pack][cle]} onChange={regler(pack, cle)} pas={pas} />
            ))}
          </div>
        ))}
      </Grille>

      <Attente etat={calcul}>
        {(m) => (
          <>
            <Legende>
              {tr(
                "Nombre total de cellules : EB = {neb}, PB = {npb}.", "Total number of cells: EB = {neb}, PB = {npb}.",
                { neb: m.nb_cellules.eb, npb: m.nb_cellules.pb }
              )}
            </Legende>
            <h2>{tr("8. Caractéristiques des packs", "8. Pack characteristics")}</h2>
            <Tableau
              colonnes={[
                { cle: "nom", titre: tr("Batterie", "Battery") },
                { cle: "tension", titre: tr("Tension (V)", "Voltage (V)"), droite: true },
                { cle: "masse", titre: tr("Masse (kg)", "Mass (kg)"), droite: true },
                { cle: "capacite", titre: tr("Capacité (Ah)", "Capacity (Ah)"), droite: true },
                { cle: "decharge", titre: tr("P décharge max (W)", "Max discharge P (W)"), droite: true },
                { cle: "recharge", titre: tr("P recharge max (W)", "Max charge P (W)"), droite: true },
                { cle: "energie", titre: tr("Énergie (Wh)", "Energy (Wh)"), droite: true },
              ]}
              lignes={m.packs.map((p) => ({
                nom: noms[p.cle], tension: nombre(p.tension_v, 1), masse: nombre(p.masse_kg, 2),
                capacite: p.capacite_ah === null ? "—" : nombre(p.capacite_ah, 2),
                decharge: nombre(p.p_decharge_w, 0), recharge: nombre(p.p_recharge_w, 0), energie: nombre(p.energie_wh, 1),
              }))}
            />
          </>
        )}
      </Attente>

      <h2>{tr("9. Convertisseur", "9. Converter")}</h2>
      <p>
        {tr(
          "La puissance du convertisseur dépend du nombre de modules installés en parallèle. Les valeurs par défaut (un module) donnent 1 520 W en décharge et −760 W en recharge.",
          "The converter power depends on the number of modules installed in parallel. The default values (one module) give 1,520 W in discharge and −760 W in charge."
        )}
      </p>
      <Grille colonnes={3}>
        <Nombre titre={tr("Nombre de modules", "Number of modules")} valeur={cellules.convertisseur.n_modules} onChange={regler("convertisseur", "n_modules")} min={1} pas={1} />
        <Nombre titre={tr("Puissance de décharge max par module (W)", "Max discharge power per module (W)")} valeur={cellules.convertisseur.p_decharge} onChange={regler("convertisseur", "p_decharge")} pas={10} />
        <Nombre titre={tr("Puissance de recharge max par module (W)", "Max charge power per module (W)")} valeur={cellules.convertisseur.p_recharge} onChange={regler("convertisseur", "p_recharge")} pas={10} />
      </Grille>
      {calcul.donnees
        ? ((m) => (
          <>
            <Grille colonnes={2}>
              <Indicateur titre={tr("Puissance totale de décharge", "Total discharge power")} valeur={`${nombre(m.convertisseur.p_decharge_w, 0)} W`} />
              <Indicateur titre={tr("Puissance totale de recharge", "Total charge power")} valeur={`${nombre(m.convertisseur.p_recharge_w, 0)} W`} />
            </Grille>
            <Legende>
              {tr(
                "Le filtre de sécurité limite la puissance traitée par le convertisseur entre {pmin} W et {pmax} W dans les simulations lancées avec ces réglages.",
                "The safety filter keeps the power processed by the converter between {pmin} W and {pmax} W in the simulations run with these settings.",
                { pmin: nombre(m.convertisseur.p_recharge_w, 0), pmax: nombre(m.convertisseur.p_decharge_w, 0) }
              )}
            </Legende>
          </>
        ))(calcul.donnees)
        : null}
    </>
  );
}

// Premier aperçu du cycle préparé avec le modèle physique.
function Apercu({ prepare, cellules }) {
  const { tr } = useLangue();
  const [soc, setSoc] = useState({ eb: 100, pb: 100 });
  const apercu = useCalcul(`/cycles-prepares/${prepare.id}/apercu`, { soc_eb0: soc.eb / 100, soc_pb0: soc.pb / 100, materiel: cellules }, 400);
  const nomEb = tr("Batterie Énergie", "Energy battery");
  const nomPb = tr("Batterie Puissance", "Power battery");
  const nomConv = tr("Convertisseur", "Converter");
  const bulle = (nom, valeur) => `${nom}${tr(" : ", ": ")}${valeur}<extra></extra>`;

  return (
    <>
      <h2>{tr("Premier aperçu avec le modèle physique", "First look with the physical model")}</h2>
      <p>
        {tr(
          "Simulation rapide avec le modèle physique (la batterie Énergie d'abord, la batterie Puissance pour le surplus), qui ne demande aucun apprentissage, avec les batteries et le convertisseur réglés ci-dessus. La comparaison des sept stratégies se lance depuis « Lancer une simulation ».",
          "Quick simulation with the physical model (the Energy battery first, the Power battery for the surplus), which needs no training, with the batteries and converter set above. The comparison of the seven strategies is run from “Run a simulation”."
        )}
      </p>
      <Grille colonnes={2}>
        <Curseur titre={tr("SOC initial de l'EB", "Initial EB SOC")} valeur={soc.eb} onChange={(v) => setSoc({ ...soc, eb: v })} min={20} max={100} unite=" %" />
        <Curseur titre={tr("SOC initial de la PB", "Initial PB SOC")} valeur={soc.pb} onChange={(v) => setSoc({ ...soc, pb: v })} min={20} max={100} unite=" %" />
      </Grille>
      <Attente etat={apercu}>
        {(a) => {
          const titres = [
            tr("États de charge (%)", "States of charge (%)"),
            tr("Puissance des batteries (kW) : décharge au-dessus de zéro, recharge en dessous", "Battery power (kW): discharge above zero, charge below"),
            tr("Courants (A)", "Currents (A)"),
            tr("Puissance traitée par le convertisseur (kW)", "Power processed by the converter (kW)"),
          ];
          const axes = ["y", "y2", "y3", "y4"];
          const courbe = (nom, y, couleur, axeY, unite, legende) => ({
            type: "scattergl", mode: "lines", x: a.temps_min, y, yaxis: axeY, name: nom, legendgroup: nom, showlegend: legende,
            line: { color: couleur, width: 1.2 }, hovertemplate: bulle(nom, `%{y:.1f} ${unite}`),
          });
          const limite = (y, axeY) => ({
            type: "line", xref: "paper", x0: 0, x1: 1, yref: axeY, y0: y, y1: y, line: { color: COULEURS.secondaire, dash: "dot", width: 1 },
          });
          return (
            <Graphique
              hauteur={960}
              courbes={[
                courbe(nomEb, a.soc_eb_pct, COULEURS.eb, "y", "%", true), courbe(nomPb, a.soc_pb_pct, COULEURS.pb, "y", "%", true),
                courbe(nomEb, a.eb_kw, COULEURS.eb, "y2", "kW", false), courbe(nomPb, a.pb_kw, COULEURS.pb, "y2", "kW", false),
                courbe(nomEb, a.eb_a, COULEURS.eb, "y3", "A", false), courbe(nomPb, a.pb_a, COULEURS.pb, "y3", "A", false),
                courbe(nomConv, a.conv_kw, COULEURS.convertisseur, "y4", "kW", false),
              ]}
              disposition={{
                hovermode: "x unified", margin: { t: 64, b: 44, l: 56, r: 16 }, legend: { orientation: "h", y: 1.07, x: 0 },
                grid: { rows: 4, columns: 1, subplots: [["xy"], ["xy2"], ["xy3"], ["xy4"]], roworder: "top to bottom", ygap: 0.22 },
                xaxis: axe({ title: { text: tr("Temps (min)", "Time (min)") }, hoverformat: ".1f" }),
                yaxis: axe(), yaxis2: axe(), yaxis3: axe(), yaxis4: axe(),
                shapes: [
                  { ...limite(a.limites.soc_min_pct, "y"), line: { color: COULEURS.violation, dash: "dot", width: 1 } },
                  limite(a.limites.p_eb_max_kw, "y2"), limite(a.limites.p_conv_max_kw, "y4"), limite(a.limites.p_conv_min_kw, "y4"),
                ],
                annotations: titres.map((titre, i) => ({
                  text: titre, xref: "paper", x: 0.5, yref: `${axes[i]} domain`, y: 1, yanchor: "bottom", showarrow: false, font: { size: 13 },
                })),
              }}
            />
          );
        }}
      </Attente>
    </>
  );
}

export default function Preparer() {
  const { tr, nombre, langue } = useLangue();
  const { prepare, setPrepare, materiel, setMateriel } = useEtat();
  const defauts = useDonnees("/preparation/defauts");
  const [fichier, setFichier] = useState(null); // analyse du fichier importé
  const [sansEntete, setSansEntete] = useState(false);
  const [erreur, setErreur] = useState(null);
  const [occupe, setOccupe] = useState(false);
  const [choix, setChoix] = useState(null); // colonnes, unité, répétitions
  const [vitesse, setVitesse] = useState(null); // unité et répétitions détectées pour la colonne de vitesse
  const [vehicule, setVehicule] = useState(null);

  useEffect(() => {
    if (defauts.donnees && !vehicule) setVehicule(defauts.donnees.vehicule);
    if (defauts.donnees && !materiel) setMateriel(defauts.donnees.materiel.cellules);
  }, [defauts.donnees, vehicule, materiel, setMateriel]);

  const adopter = (analyse) => {
    setFichier(analyse);
    setVitesse(analyse.vitesse);
    setChoix({
      col_vitesse: analyse.devine.speed || AUCUNE, col_puissance: analyse.devine.power || AUCUNE,
      col_acceleration: analyse.devine.acceleration || AUCUNE, col_temps: analyse.devine.time || analyse.colonnes[0],
      sans_temps: analyse.colonnes.length === 1 || !analyse.devine.time, frequence_hz: 1,
      unite_vitesse: (analyse.vitesse && analyse.vitesse.unite) || "km/h", repetitions: (analyse.vitesse && analyse.vitesse.repetitions) || 1,
    });
  };

  const importer = async (contenu, sans) => {
    setErreur(null);
    setOccupe(true);
    try {
      adopter(await envoyerFichier("/fichiers", { fichier: contenu, sans_entete: sans }, langue));
    } catch (e) {
      setErreur(e.message);
      setFichier(null);
    }
    setOccupe(false);
  };

  const relire = async (options) => {
    setErreur(null);
    try {
      adopter(await envoyer(`/fichiers/${fichier.id}/relire`, options, langue));
    } catch (e) {
      setErreur(e.message);
    }
  };

  const choisirVitesse = async (colonne) => {
    setChoix((c) => ({ ...c, col_vitesse: colonne }));
    if (!colonne) return setVitesse(null);
    try {
      const analyse = await lire(`/fichiers/${fichier.id}/vitesse`, { colonne, sans_entete: sansEntete, feuille: fichier.feuille }, langue);
      setVitesse(analyse);
      if (analyse.numerique) setChoix((c) => ({ ...c, unite_vitesse: analyse.unite, repetitions: analyse.repetitions }));
    } catch (e) {
      setErreur(e.message);
    }
    return undefined;
  };

  const preparer = async () => {
    setErreur(null);
    setOccupe(true);
    try {
      setPrepare(await envoyer("/cycles-prepares", {
        id_fichier: fichier.id, sans_entete: sansEntete, feuille: fichier.feuille,
        col_vitesse: choix.col_vitesse || null, col_puissance: choix.col_puissance || null,
        col_acceleration: choix.col_acceleration || null, col_temps: choix.sans_temps ? null : choix.col_temps,
        frequence_hz: choix.frequence_hz, unite_vitesse: choix.unite_vitesse, repetitions: choix.repetitions, vehicule,
      }, langue));
    } catch (e) {
      setErreur(e.message);
    }
    setOccupe(false);
  };

  const regler = (cle) => (valeur) => setChoix((c) => ({ ...c, [cle]: valeur }));
  const colonnes = fichier ? fichier.colonnes.map((c) => ({ valeur: c, nom: c })) : [];
  const avecAucune = (texte) => [{ valeur: AUCUNE, nom: texte }, ...colonnes];
  const vitesseValide = !choix || !choix.col_vitesse || (vitesse && vitesse.numerique);

  return (
    <>
      <h1>{tr("📂 Préparer une simulation", "📂 Prepare a simulation")}</h1>
      <p>
        {tr(
          "Importez un fichier de cycle de conduite au format CSV, TXT, TSV ou Excel, avec ou sans ligne d'en-tête. Les colonnes de temps, de vitesse, de puissance et d'accélération sont reconnues d'après leur nom, en français ou en anglais ; vous pouvez ensuite corriger ces choix.",
          "Import a driving cycle file in CSV, TXT, TSV or Excel format, with or without a header row. The time, speed, power and acceleration columns are recognised from their names, in French or English; you can then correct these choices."
        )}
      </p>

      <h2>{tr("1. Importation du fichier", "1. File import")}</h2>
      <Case
        titre={tr(
          "Ce fichier ne contient qu'une seule colonne : le profil de vitesse brut, sans ligne d'en-tête",
          "This file contains a single column: the raw speed profile, without a header row"
        )}
        coche={sansEntete}
        onChange={(v) => {
          setSansEntete(v);
          if (fichier) relire({ sans_entete: v, feuille: fichier.feuille });
        }}
      />
      <label className="champ">
        <span className="champ-titre">{tr("Fichier du cycle de conduite", "Driving cycle file")}</span>
        <input type="file" accept=".csv,.txt,.tsv,.xlsx,.xls" onChange={(e) => e.target.files[0] && importer(e.target.files[0], sansEntete)} />
      </label>
      <Message type="erreur">{erreur}</Message>
      {!fichier ? <Message type="info">{occupe ? tr("Lecture du fichier…", "Reading the file…") : tr("En attente d'un fichier.", "Waiting for a file.")}</Message> : null}

      {fichier && choix ? (
        <>
          {fichier.feuilles && fichier.feuilles.length > 1 ? (
            <Selection
              titre={tr("Feuille à utiliser", "Sheet to use")} valeur={fichier.feuille}
              onChange={(feuille) => relire({ sans_entete: sansEntete, feuille })} options={fichier.feuilles.map((f) => ({ valeur: f, nom: f }))}
            />
          ) : null}
          {fichier.colonnes_sans_nom ? (
            <Message type="alerte">
              {tr(
                "Les colonnes de ce fichier n'ont pas de nom : l'en-tête est absent ou n'a pas été reconnu, elles sont donc numérotées 0, 1, 2… S'il s'agit d'un profil de vitesse brut à une seule colonne, cochez la case ci-dessus.",
                "The columns of this file have no name: the header is missing or was not recognised, so they are numbered 0, 1, 2… If it is a raw single-column speed profile, tick the box above."
              )}
            </Message>
          ) : null}
          <Message type="succes">
            {tr("Fichier chargé : {l} lignes et {c} colonnes.", "File loaded: {l} rows and {c} columns.", { l: fichier.nb_lignes, c: fichier.colonnes.length })}
          </Message>
          <Depliant titre={tr("Aperçu des données brutes", "Raw data preview")}>
            <Tableau
              hauteur={360}
              colonnes={fichier.colonnes.map((c) => ({ cle: c, titre: c, droite: true }))}
              lignes={fichier.apercu.map((ligne) => Object.fromEntries(fichier.colonnes.map((c) => [c, ligne[c] === null ? "" : String(ligne[c])])))}
            />
          </Depliant>

          <h2>{tr("2. Sélection des colonnes", "2. Column selection")}</h2>
          <Grille colonnes={2}>
            <Selection titre={tr("Colonne de vitesse", "Speed column")} valeur={choix.col_vitesse} onChange={choisirVitesse} options={avecAucune(tr("(aucune)", "(none)"))} />
            <Selection
              titre={tr("Colonne de puissance demandée", "Power demand column")} valeur={choix.col_puissance} onChange={regler("col_puissance")}
              options={avecAucune(tr("(aucune — calculer à partir de la dynamique du véhicule)", "(none — compute from the vehicle dynamics)"))}
            />
            <Selection
              titre={tr("Colonne d'accélération (facultative)", "Acceleration column (optional)")} valeur={choix.col_acceleration}
              onChange={regler("col_acceleration")} options={avecAucune(tr("(aucune — considérée comme nulle)", "(none — taken as zero)"))}
            />
            <Case
              titre={tr("Aucune colonne de temps (échantillonnage à fréquence constante)", "No time column (constant sampling rate)")}
              coche={choix.sans_temps} onChange={regler("sans_temps")}
            />
            <Nombre
              titre={tr("Fréquence d'échantillonnage (Hz), utilisée en l'absence de colonne de temps", "Sampling rate (Hz), used when there is no time column")}
              valeur={choix.frequence_hz} onChange={regler("frequence_hz")} min={0.01} pas={0.1}
              aide={tr("1 Hz = un point par seconde (cas des cycles WLTC et Artemis).", "1 Hz = one point per second (as for the WLTC and Artemis cycles).")}
            />
            <Selection
              titre={tr("Colonne de temps, utilisée si le fichier en contient une", "Time column, used if the file has one")}
              valeur={choix.col_temps} onChange={regler("col_temps")} options={colonnes}
            />
          </Grille>

          <h2>{tr("3. Unité de la vitesse", "3. Speed unit")}</h2>
          {choix.col_vitesse ? (
            <>
              {vitesse && vitesse.numerique ? (
                <Legende>
                  {tr(
                    "Unité reconnue automatiquement : **{u}** (vitesse maximale ≈ {v} dans l'unité d'origine).",
                    "Automatically recognised unit: **{u}** (maximum speed ≈ {v} in the original unit).",
                    { u: vitesse.unite, v: nombre(vitesse.maximum, 1) }
                  )}
                </Legende>
              ) : (
                <Message type="erreur">
                  {tr("La colonne de vitesse « {c} » ne contient pas de nombres.", "The speed column “{c}” contains no numbers.", { c: choix.col_vitesse })}
                </Message>
              )}
              <Radios
                titre={tr("Unité de la colonne de vitesse", "Unit of the speed column")} valeur={choix.unite_vitesse} onChange={regler("unite_vitesse")}
                options={[{ valeur: "km/h", nom: "km/h" }, { valeur: "m/s", nom: "m/s" }]}
              />
            </>
          ) : (
            <Legende>{tr("Aucune colonne de vitesse sélectionnée : cette section est ignorée.", "No speed column selected: this section is skipped.")}</Legende>
          )}

          <h2>{tr("4. Répétition du cycle", "4. Cycle repetition")}</h2>
          {vitesse && vitesse.numerique ? (
            <Legende>
              {tr(
                "Détection automatique, à titre indicatif : le motif semble répété {n} fois.",
                "Automatic detection, for information: the pattern seems to be repeated {n} times.",
                { n: vitesse.repetitions }
              )}
            </Legende>
          ) : null}
          <Grille colonnes={2}>
            <Nombre
              titre={tr("Nombre de répétitions du cycle à appliquer", "Number of cycle repetitions to apply")} valeur={choix.repetitions}
              onChange={regler("repetitions")} min={1} max={50} pas={1}
              aide={tr(
                "Indiquez la valeur exacte connue pour votre fichier plutôt que de vous fier uniquement à la détection automatique.",
                "Enter the exact value known for your file rather than relying only on the automatic detection."
              )}
            />
          </Grille>

          <h2>{tr("5. Paramètres du véhicule (utilisés seulement sans colonne de puissance)", "5. Vehicle parameters (used only without a power column)")}</h2>
          <Legende>
            {tr(
              "La puissance est calculée à partir des forces aérodynamique, de roulement, de gravité et d'inertie. Les valeurs par défaut sont celles du véhicule de l'article de référence.",
              "Power is computed from the aerodynamic, rolling, gravity and inertia forces. The default values are those of the vehicle in the reference paper."
            )}
          </Legende>
          {vehicule ? (
            <Grille colonnes={3}>
              {[
                ["masse", tr("Masse (kg)", "Mass (kg)"), 10], ["cx", tr("Coefficient de traînée Cx", "Drag coefficient Cx"), 0.01],
                ["surface", tr("Surface frontale S (m²)", "Frontal area S (m²)"), 0.05], ["c0", tr("Coefficient de roulement C0", "Rolling coefficient C0"), 0.001],
                ["c1", tr("Coefficient de roulement C1", "Rolling coefficient C1"), 0.0000001], ["pente_deg", tr("Pente de la route (°)", "Road slope (°)"), 0.5],
                ["rho", tr("Masse volumique de l'air (kg/m³)", "Air density (kg/m³)"), 0.005],
              ].map(([cle, titre, pas]) => (
                <Nombre key={cle} titre={titre} valeur={vehicule[cle]} onChange={(v) => setVehicule({ ...vehicule, [cle]: v })} pas={pas} />
              ))}
            </Grille>
          ) : null}
          <p>
            <button className="bouton principal" onClick={preparer} disabled={occupe || !vitesseValide || (!choix.col_vitesse && !choix.col_puissance)}>
              {occupe ? tr("Préparation en cours…", "Preparing…") : tr("Préparer les données", "Prepare the data")}
            </button>
          </p>
        </>
      ) : null}

      {prepare && materiel ? (
        <>
          <Message type="succes">{tr("Données préparées : {n} points au total.", "Data prepared: {n} points in total.", { n: prepare.nb_points })}</Message>
          <hr className="separateur" />

          <h2>{tr("6. Formules utilisées", "6. Formulas used")}</h2>
          <Depliant titre={tr("Dynamique du véhicule et packs de batteries", "Vehicle dynamics and battery packs")}>
            <p>
              <strong>{tr("Dynamique longitudinale du véhicule", "Longitudinal vehicle dynamics")}</strong>{" "}
              {tr("(si la puissance n'est pas fournie) :", "(if the power is not provided):")}
            </p>
            <Formule tex={"a = \\frac{dv}{dt} \\qquad F_{aero} = \\tfrac{1}{2}\\,\\rho\\,S\\,C_x\\,v^2 \\qquad F_{roul} = m\\,g\\,(C_0 + C_1 v^2) \\qquad F_{grav} = m\\,g\\,\\sin\\theta"} />
            <Formule tex={"F_{tot} = F_{aero} + F_{roul} + F_{grav} + m\\,a \\qquad P_{dem} = F_{tot}\\,v"} />
            <p>
              <strong>{tr("Packs de batteries", "Battery packs")}</strong> {tr("(à partir des cellules, sections 7 et 8) :", "(from the cells, sections 7 and 8):")}
            </p>
            <Formule tex={"V = V_{cell}\\,n_s \\qquad M = m_{cell}\\,n_s\\,n_p \\qquad P_{max} = I_{cell}\\,V\\,n_p \\qquad E = DE \\times M \\qquad C = C_{cell}\\,n_p"} />
          </Depliant>

          <Materiel cellules={materiel} setCellules={setMateriel} />

          <hr className="separateur" />
          <Grille colonnes={3}>
            <Indicateur titre={tr("Nombre de points", "Number of points")} valeur={nombre(prepare.nb_points, 0)} />
            <Indicateur titre={tr("Répétitions", "Repetitions")} valeur={prepare.repetitions} />
            <Indicateur
              titre={tr("Origine de la puissance", "Origin of the power")}
              valeur={prepare.puissance_calculee ? tr("calculée (dynamique du véhicule)", "computed (vehicle dynamics)") : tr("colonne du fichier", "file column")}
            />
          </Grille>
          {prepare.puissance_presque_nulle ? (
            <Message type="alerte">
              {tr(
                "La puissance moyenne calculée est presque nulle (< 50 W). Vérifiez l'unité de vitesse, la colonne choisie et la fréquence d'échantillonnage ; sinon, le SOC variera très peu.",
                "The computed mean power is almost zero (< 50 W). Check the speed unit, the selected column and the sampling rate; otherwise the SOC will barely change."
              )}
            </Message>
          ) : null}

          <h2>{tr("Télécharger les données préparées", "Download the prepared data")}</h2>
          <p>
            {tr(
              "Le fichier contient le temps, la vitesse, l'accélération, les forces et la puissance demandée.",
              "The file contains the time, speed, acceleration, forces and power demand."
            )}
          </p>
          <div className="rangee">
            <a className="bouton etroit" href={`/api/cycles-prepares/${prepare.id}/export?format=xlsx`}>
              {tr("Télécharger (Excel)", "Download (Excel)")}
            </a>
            <a className="bouton etroit" href={`/api/cycles-prepares/${prepare.id}/export?format=csv`}>
              {tr("Télécharger (CSV)", "Download (CSV)")}
            </a>
            <span />
          </div>

          <h2>{tr("Aperçu du cycle", "Cycle overview")}</h2>
          <Grille colonnes={2}>
            <div>
              <h4>{prepare.apercu.vitesse_kmh ? tr("Vitesse (km/h)", "Speed (km/h)") : tr("Accélération (m/s²)", "Acceleration (m/s²)")}</h4>
              <Graphique
                hauteur={300}
                courbes={[{
                  type: "scattergl", mode: "lines", x: prepare.apercu.temps_min, y: prepare.apercu.vitesse_kmh || prepare.apercu.acceleration,
                  line: { color: COULEURS.demande, width: 1.2 }, hovertemplate: "%{y:.1f}<extra></extra>",
                }]}
                disposition={{ showlegend: false, hovermode: "x unified", xaxis: axe({ title: { text: tr("Temps (min)", "Time (min)") }, hoverformat: ".1f" }) }}
              />
            </div>
            <div>
              <h4>{tr("Puissance demandée (kW)", "Power demand (kW)")}</h4>
              <Graphique
                hauteur={300}
                courbes={[{
                  type: "scattergl", mode: "lines", x: prepare.apercu.temps_min, y: prepare.apercu.puissance_kw,
                  line: { color: COULEURS.demande, width: 1.2 }, hovertemplate: "%{y:.1f} kW<extra></extra>",
                }]}
                disposition={{ showlegend: false, hovermode: "x unified", xaxis: axe({ title: { text: tr("Temps (min)", "Time (min)") }, hoverformat: ".1f" }) }}
              />
            </div>
          </Grille>

          <Apercu prepare={prepare} cellules={materiel} />

          <p style={{ marginTop: 24 }}>
            <Link className="bouton principal" to="/lancer">
              {tr("Lancer une simulation sur ce cycle →", "Run a simulation on this cycle →")}
            </Link>
          </p>
        </>
      ) : null}

      <PiedNavigation page="/preparer" />
    </>
  );
}
