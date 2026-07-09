
"""
test_ems_core.py

Ce script vérifie le comportement du système de gestion d’énergie
dans plusieurs situations physiques représentatives.

La valeur alpha proposée est toujours calculée par la stratégie
EB-priority. Elle n’est jamais choisie manuellement.

Exécution :
    python code/test_ems_core.py
"""

from ems_core import (
    resoudre_decision_physique,
    eb_priority_alpha_single,
)


SITUATIONS = [
    (
        "Véhicule à l'arrêt",
        dict(p_dem=0.0, soc_eb=0.60, soc_pb=0.60),
    ),
    (
        "Faible demande de traction",
        dict(p_dem=2000.0, soc_eb=0.70, soc_pb=0.70),
    ),
    (
        "Demande supérieure à la limite de l'EB",
        dict(p_dem=30000.0, soc_eb=0.80, soc_pb=0.80),
    ),
    (
        "Forte traction avec batteries faibles",
        dict(p_dem=40000.0, soc_eb=0.20, soc_pb=0.20),
    ),
    (
        "Freinage régénératif normal",
        dict(p_dem=-10000.0, soc_eb=0.60, soc_pb=0.60),
    ),
    (
        "Freinage avec batteries presque pleines",
        dict(p_dem=-45000.0, soc_eb=0.99, soc_pb=0.999),
    ),
]


def formater_statut_faisabilite(feasible):
    return "Oui" if feasible else "Non"


def formater_resultat_bilan(ok, ecart):
    if ok:
        return "Le bilan de puissance est respecté."

    return (
        "Une incohérence a été détectée dans le bilan de puissance "
        f"(écart : {ecart:.6f} W)."
    )


def formater_alpha(alpha, p_eb, p_pb):
    puissance_repartie = abs(p_eb) + abs(p_pb)

    if puissance_repartie <= 1e-9:
        return "Non applicable"

    return f"{alpha:.3f}"


def formater_nombre(nombre, singulier, pluriel):
    if nombre == 1:
        return f"1 {singulier}"

    return f"{nombre} {pluriel}"


print()
print("Vérification du module physique de gestion d’énergie")
print()
print(
    "Les situations ci-dessous sont uniquement des cas de test. "
    "Elles permettent de vérifier que le module physique réagit "
    "correctement avant son utilisation sur un cycle de conduite complet."
)
print()

tous_les_bilans_corrects = True
resultats = []

for numero, (nom, params) in enumerate(SITUATIONS, start=1):
    p_dem = params["p_dem"]
    soc_eb = params["soc_eb"]
    soc_pb = params["soc_pb"]

    alpha_requested = eb_priority_alpha_single(
        p_dem=p_dem,
        soc_eb=soc_eb,
    )

    decision = resoudre_decision_physique(
        alpha_requested=alpha_requested,
        p_dem=p_dem,
        soc_eb=soc_eb,
        soc_pb=soc_pb,
        alpha_prev=None,
    )

    bilan_puissance = (
        decision["P_EB_final"]
        + decision["P_PB_final"]
        + decision["P_unserved"]
        - decision["P_regen_curtailed"]
    )

    ecart = p_dem - bilan_puissance
    bilan_correct = abs(ecart) < 1e-6

    tous_les_bilans_corrects = (
        tous_les_bilans_corrects and bilan_correct
    )

    statut_faisabilite = formater_statut_faisabilite(
        decision["feasible"]
    )

    statut_bilan = formater_resultat_bilan(
        bilan_correct,
        ecart,
    )

    alpha_propose_affiche = formater_alpha(
        alpha_requested,
        decision["P_EB_final"],
        decision["P_PB_final"],
    )

    alpha_applique_affiche = formater_alpha(
        decision["alpha_final"],
        decision["P_EB_final"],
        decision["P_PB_final"],
    )

    resultats.append(
        {
            "situation": nom,
            "feasible": decision["feasible"],
            "bilan_correct": bilan_correct,
            "P_unserved": decision["P_unserved"],
            "P_regen_curtailed": decision["P_regen_curtailed"],
        }
    )

    print(f"Scénario {numero} : {nom}")
    print(f"Demande de puissance : {p_dem:.1f} W")
    print(f"SOC initial de l'EB : {soc_eb:.3f}")
    print(f"SOC initial de la PB : {soc_pb:.3f}")
    print(f"Alpha proposé par EB-priority : {alpha_propose_affiche}")
    print(f"Alpha réellement appliqué : {alpha_applique_affiche}")
    print(
        f"Puissance prise en charge par l'EB : "
        f"{decision['P_EB_final']:.1f} W"
    )
    print(
        f"Puissance prise en charge par la PB : "
        f"{decision['P_PB_final']:.1f} W"
    )

    if decision["P_unserved"] > 1e-9:
        print(
            f"Puissance de traction non servie : "
            f"{decision['P_unserved']:.1f} W"
        )

    if decision["P_regen_curtailed"] > 1e-9:
        print(
            f"Puissance de freinage non récupérée : "
            f"{decision['P_regen_curtailed']:.1f} W"
        )

    print(
        f"Demande entièrement prise en charge : "
        f"{statut_faisabilite}"
    )
    print(f"Vérification : {statut_bilan}")

    if "explanation" in decision:
        print(f"Interprétation : {decision['explanation']}")

    print()


nb_scenarios = len(resultats)

nb_non_faisables = sum(
    not resultat["feasible"]
    for resultat in resultats
)

nb_avec_puissance_non_servie = sum(
    resultat["P_unserved"] > 1e-9
    for resultat in resultats
)

nb_avec_regeneration_rejetee = sum(
    resultat["P_regen_curtailed"] > 1e-9
    for resultat in resultats
)


print("Synthèse générale")
print()

if tous_les_bilans_corrects:
    print(
        "Le bilan de puissance est correctement respecté "
        "dans tous les scénarios testés."
    )

    if nb_non_faisables == 0:
        print(
            "Toutes les demandes peuvent être entièrement prises "
            "en charge par le système HESS."
        )
    else:
        situations_non_faisables = formater_nombre(
            nb_non_faisables,
            "situation ne peut",
            "situations ne peuvent",
        )

        print(
            f"Parmi les {nb_scenarios} situations testées, "
            f"{situations_non_faisables} pas être entièrement prises "
            "en charge en raison des limites physiques du système."
        )

        if nb_avec_puissance_non_servie > 0:
            situations_non_servies = formater_nombre(
                nb_avec_puissance_non_servie,
                "situation présente",
                "situations présentent",
            )

            print(
                f"{situations_non_servies} une demande de traction "
                "qui ne peut pas être entièrement fournie."
            )

        if nb_avec_regeneration_rejetee > 0:
            situations_regeneration = formater_nombre(
                nb_avec_regeneration_rejetee,
                "situation présente",
                "situations présentent",
            )

            print(
                f"{situations_regeneration} une énergie de freinage "
                "qui ne peut pas être entièrement récupérée."
            )

        print(
            "Dans ces cas, la puissance non servie ou l’énergie "
            "de freinage non récupérée est correctement comptabilisée."
        )

    print()
    print(
        "Conclusion : le module physique est cohérent sur ces cas de test "
        "et peut maintenant être utilisé sur un cycle de conduite complet."
    )

else:
    nb_bilans_incorrects = sum(
        not resultat["bilan_correct"]
        for resultat in resultats
    )

    texte_bilans_incorrects = formater_nombre(
        nb_bilans_incorrects,
        "bilan est incorrect",
        "bilans sont incorrects",
    )

    print(
        "Au moins une incohérence a été détectée dans le bilan de puissance."
    )
    print(
        f"Au total, {texte_bilans_incorrects} "
        f"sur les {nb_scenarios} scénarios testés."
    )
    print(
        "Les situations concernées doivent être vérifiées avant de lancer "
        "une simulation sur un cycle complet."
    )

print()

