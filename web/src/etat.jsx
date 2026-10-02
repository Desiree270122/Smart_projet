// État partagé par les pages : le cycle affiché, l'instant étudié, la dernière
// simulation lancée, le cycle préparé et le matériel (batteries, convertisseur).
// Le choix du cycle et la dernière simulation sont gardés d'une visite à l'autre.

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { lire, useDonnees } from "./api.js";
import { useLangue } from "./langue.jsx";

const Contexte = createContext(null);
const CLE_CYCLE = "2smart.cycle";
const CLE_SIMULATION = "2smart.simulation";

export function FournisseurEtat({ children }) {
  const { langue, tr } = useLangue();
  const meta = useDonnees("/meta");
  const [cycle, setCycle] = useState(() => localStorage.getItem(CLE_CYCLE) || "artemis");
  const [simulation, setSimulation] = useState(() => localStorage.getItem(CLE_SIMULATION));
  const [instant, setInstant] = useState(null); // en secondes ; null = milieu du cycle
  const [prepare, setPrepare] = useState(null); // cycle préparé : { id, nb_points, … }
  const [materiel, setMateriel] = useState(null); // réglages des cellules et du convertisseur ; null = ceux du projet

  const choisirCycle = useCallback((identifiant) => {
    localStorage.setItem(CLE_CYCLE, identifiant);
    setCycle(identifiant);
  }, []);

  const adopterSimulation = useCallback(
    (identifiant) => {
      localStorage.setItem(CLE_SIMULATION, identifiant);
      setSimulation(identifiant);
      choisirCycle(identifiant);
    },
    [choisirCycle]
  );

  // La dernière simulation vit dans la mémoire du serveur : si elle n'y est plus
  // (serveur relancé), on revient au cycle de référence.
  useEffect(() => {
    if (!simulation) return;
    lire(`/simulations/${simulation}`, null, langue, { garder: false })
      .then((etat) => {
        if (etat.etat !== "terminee") throw new Error("simulation indisponible");
      })
      .catch(() => {
        localStorage.removeItem(CLE_SIMULATION);
        setSimulation(null);
        setCycle((courant) => (courant === simulation ? "artemis" : courant));
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [simulation]);

  const valeur = useMemo(() => {
    const cycles = [...((meta.donnees && meta.donnees.cycles) || [])];
    if (simulation) cycles.push({ id: simulation, nom: tr("Ma dernière simulation", "My last simulation") });
    const connu = cycles.some((c) => c.id === cycle);
    return {
      meta: meta.donnees,
      erreurServeur: meta.erreur,
      cycles,
      cycle: connu || cycles.length === 0 ? cycle : cycles[0].id,
      choisirCycle,
      simulation,
      adopterSimulation,
      instant,
      setInstant,
      prepare,
      setPrepare,
      materiel,
      setMateriel,
    };
  }, [meta.donnees, meta.erreur, cycle, simulation, instant, prepare, materiel, choisirCycle, adopterSimulation, tr]);

  return <Contexte.Provider value={valeur}>{children}</Contexte.Provider>;
}

export function useEtat() {
  return useContext(Contexte);
}
