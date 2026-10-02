// Appels au serveur de calcul (dossier api/). Toutes les routes sont sous /api
// et acceptent ?lang=fr|en : les phrases rédigées reviennent dans cette langue.

import { useEffect, useRef, useState } from "react";
import { useLangue } from "./langue.jsx";

const memoire = new Map(); // réponses déjà reçues, par adresse

function adresse(chemin, parametres, langue) {
  const requete = new URLSearchParams();
  Object.entries(parametres || {}).forEach(([cle, valeur]) => {
    if (valeur !== undefined && valeur !== null) requete.set(cle, valeur);
  });
  requete.set("lang", langue);
  return `/api${chemin}?${requete.toString()}`;
}

async function repondre(reponse) {
  let contenu = null;
  try {
    contenu = await reponse.json();
  } catch {
    contenu = null;
  }
  if (!reponse.ok) {
    const erreur = new Error((contenu && contenu.erreur) || `Erreur ${reponse.status}`);
    erreur.code = reponse.status;
    throw erreur;
  }
  return contenu;
}

export async function lire(chemin, parametres, langue, { garder = true } = {}) {
  const url = adresse(chemin, parametres, langue);
  if (garder && memoire.has(url)) return memoire.get(url);
  const contenu = await repondre(await fetch(url));
  if (garder) memoire.set(url, contenu);
  return contenu;
}

export async function envoyer(chemin, corps, langue) {
  return repondre(
    await fetch(adresse(chemin, null, langue), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(corps || {}),
    })
  );
}

export async function envoyerFichier(chemin, champs, langue) {
  const formulaire = new FormData();
  Object.entries(champs).forEach(([cle, valeur]) => {
    if (valeur !== undefined && valeur !== null) formulaire.append(cle, valeur);
  });
  return repondre(await fetch(adresse(chemin, null, langue), { method: "POST", body: formulaire }));
}

export function oublier(prefixe) {
  [...memoire.keys()].filter((url) => url.includes(prefixe)).forEach((url) => memoire.delete(url));
}

// Données d'une route : { donnees, erreur, chargement }. Rien n'est demandé tant
// que `chemin` est vide (par exemple quand la stratégie n'est pas encore choisie).
export function useDonnees(chemin, parametres, options) {
  const { langue } = useLangue();
  const cle = chemin ? adresse(chemin, parametres, langue) : null;
  const [etat, setEtat] = useState({ donnees: null, erreur: null, chargement: Boolean(cle), cle: null });

  useEffect(() => {
    if (!cle) {
      setEtat({ donnees: null, erreur: null, chargement: false, cle: null });
      return undefined;
    }
    let actif = true;
    setEtat((precedent) => ({ donnees: precedent.cle === cle ? precedent.donnees : null, erreur: null, chargement: true, cle }));
    lire(chemin, parametres, langue, options)
      .then((donnees) => actif && setEtat({ donnees, erreur: null, chargement: false, cle }))
      .catch((erreur) => actif && setEtat({ donnees: null, erreur, chargement: false, cle }));
    return () => {
      actif = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cle]);

  // Tant que la réponse à la demande courante n'est pas arrivée, on ne montre pas celle de la
  // précédente — sauf avec l'option « conserver », qui la laisse affichée pendant l'attente
  // (utile quand on fait glisser un curseur).
  const derniere = useRef(null);
  const courant = etat.cle === cle ? etat : { donnees: null, erreur: null, chargement: Boolean(cle), cle };
  if (courant.donnees) derniere.current = courant.donnees;
  if (options && options.conserver && cle && !courant.donnees && !courant.erreur && derniere.current) {
    return { ...courant, donnees: derniere.current, chargement: true };
  }
  return courant;
}

// Calcul demandé au serveur avec des réglages (envoyés en POST) : relancé, après un court
// délai, chaque fois que les réglages changent. Rien n'est demandé tant que `chemin` est vide.
export function useCalcul(chemin, reglages, delai = 300) {
  const { langue } = useLangue();
  const cle = chemin ? `${chemin}|${langue}|${JSON.stringify(reglages)}` : null;
  const [etat, setEtat] = useState({ donnees: null, erreur: null, chargement: Boolean(cle) });

  useEffect(() => {
    if (!cle) {
      setEtat({ donnees: null, erreur: null, chargement: false });
      return undefined;
    }
    let actif = true;
    setEtat((precedent) => ({ ...precedent, erreur: null, chargement: true }));
    const minuteur = setTimeout(() => {
      envoyer(chemin, reglages, langue)
        .then((donnees) => actif && setEtat({ donnees, erreur: null, chargement: false }))
        .catch((erreur) => actif && setEtat({ donnees: null, erreur, chargement: false }));
    }, delai);
    return () => {
      actif = false;
      clearTimeout(minuteur);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cle]);

  return etat;
}
