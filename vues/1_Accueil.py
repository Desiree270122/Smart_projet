from datetime import date

import plotly.graph_objects as go
import streamlit as st

import ems_core as core
from core.resultats import charger_reference


# Palette de la charte.
C_BLEU = "#3B82F6"
C_VERT = "#22C55E"
C_ORANGE = "#F59E0B"
C_GRIS = "#6B7280"
C_ARDOISE = "#334155"


st.markdown(
    """
    <style>
    .s2-topbar{display:flex;justify-content:space-between;align-items:center;
      padding:.55rem .95rem;border:1px solid rgba(128,128,128,.25);border-radius:10px;
      font-size:.82rem;color:#8B93A7;margin-bottom:1.1rem;flex-wrap:wrap;gap:6px}
    .s2-topbar b{color:#3B82F6}
    .s2-hero-title{font-size:3.2rem;font-weight:800;letter-spacing:-1px;line-height:1;
      color:#3B82F6;background:linear-gradient(90deg,#3B82F6,#22C55E);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0}
    .s2-hero-claim{font-size:1.3rem;color:#CBD5E1;margin:.6rem 0 .35rem;font-weight:600;line-height:1.35}
    .s2-hero-tags{font-size:.95rem;color:#3B82F6;font-weight:700;letter-spacing:.6px}
    .s2-kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:.2rem 0}
    .s2-kpi{border:1px solid rgba(128,128,128,.22);border-radius:14px;padding:16px 18px;
      background:rgba(127,127,127,.06)}
    .s2-kpi .n{font-size:2rem;font-weight:800;line-height:1}
    .s2-kpi .l{font-size:.85rem;color:#94A3B8;margin-top:6px}
    .s2-pipe{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin:.2rem 0 .4rem}
    .s2-step{border:1px solid rgba(128,128,128,.25);border-radius:10px;padding:8px 13px;
      font-weight:600;font-size:.9rem;background:rgba(59,130,246,.08)}
    .s2-arrow{color:#22C55E;font-weight:800;font-size:1.1rem}
    .s2-card{border:1px solid rgba(128,128,128,.22);border-radius:14px;padding:16px 18px;
      background:rgba(127,127,127,.05);height:100%}
    .s2-card .t{font-weight:700;font-size:1rem;margin-bottom:4px}
    .s2-card .d{color:#94A3B8;font-size:.9rem}
    @media(max-width:900px){.s2-kpi-grid{grid-template-columns:repeat(2,1fr)}
      .s2-hero-title{font-size:2.4rem}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _kpis():
    """KPI réels lus depuis le fichier précalculé, avec repli sûr si absent."""
    n_pts = 0
    try:
        d = charger_reference()
        strategies = list(d["resultats"].keys())
        n_pts = int(d["meta"].get("nb_points", 0))
    except Exception:  # noqa: BLE001
        strategies = list(core.MODEL_ORDER)
    n_strat = len(strategies)
    non_ia = {"EMS_power_limitation", "EMS_fuzzy_logic"}
    n_ia = sum(1 for s in strategies if s not in non_ia)
    return n_strat, n_ia, n_pts


def _schema_hess():
    """Schéma du système hybride de stockage (chaîne véhicule -> batteries)."""
    pos = {
        "Véhicule": (0.0, 3.0),
        "Moteur": (0.0, 2.0),
        "Convertisseur": (0.0, 1.0),
        "Batterie Énergie": (-1.15, 0.0),
        "Batterie Puissance": (1.15, 0.0),
    }
    couleurs = {
        "Véhicule": C_ARDOISE,
        "Moteur": C_ORANGE,
        "Convertisseur": C_BLEU,
        "Batterie Énergie": C_BLEU,
        "Batterie Puissance": C_VERT,
    }
    aretes = [
        ("Véhicule", "Moteur"),
        ("Moteur", "Convertisseur"),
        ("Convertisseur", "Batterie Énergie"),
        ("Convertisseur", "Batterie Puissance"),
    ]

    edge_x, edge_y = [], []
    for a, b in aretes:
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#C7CCD6", width=2), hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in pos.values()],
            y=[p[1] for p in pos.values()],
            mode="markers+text",
            marker=dict(size=42, color=list(couleurs.values()), line=dict(color="white", width=2)),
            text=list(pos.keys()),
            textposition="middle right",
            textfont=dict(size=12),
            hoverinfo="text",
        )
    )
    fig.update_layout(
        height=360,
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(visible=False, range=[-2.0, 3.2]),
        yaxis=dict(visible=False, range=[-0.6, 3.4]),
    )
    return fig


# Barre supérieure

st.markdown(
    f"""
    <div class="s2-topbar">
      <span><b>2SMART</b> &nbsp;·&nbsp; Plateforme HESS</span>
      <span>Version 2.0 &nbsp;·&nbsp; Exécution {str(core.DEVICE).upper()} &nbsp;·&nbsp; {date.today().isoformat()}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# Hero — une phrase qui donne immédiatement le sens du projet

hero_g, hero_d = st.columns([1.25, 1])
with hero_g:
    st.markdown(
        """
        <div class="s2-hero-title">2SMART</div>
        <div class="s2-hero-claim">Optimiser la répartition d'énergie entre deux batteries
        hybrides grâce à l'IA explicable.</div>
        <div class="s2-hero-tags">Expliquer &nbsp;·&nbsp; Simuler &nbsp;·&nbsp; Comparer &nbsp;·&nbsp; Optimiser</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("🚀 Explorer une démonstration", type="primary", use_container_width=True):
            st.switch_page("vues/5_Comparaison_des_strategies.py")
    with b2:
        if st.button("🧪 Nouvelle simulation", use_container_width=True):
            st.switch_page("vues/8_Simulation_cycle_personnalise.py")

with hero_d:
    st.plotly_chart(_schema_hess(), use_container_width=True)


# Le système, composant par composant (interactif)

st.caption("Cliquez sur un composant pour comprendre son rôle dans le système.")

comp1, comp2, comp3, comp4 = st.columns(4)
with comp1:
    with st.popover("🔋 Batterie Énergie", use_container_width=True):
        st.markdown(
            "**Batterie Énergie (EB)**\n\n"
            "- Grande capacité de stockage\n"
            "- Faible densité de puissance\n"
            "- Assure l'autonomie du véhicule"
        )
with comp2:
    with st.popover("⚡ Batterie Puissance", use_container_width=True):
        st.markdown(
            "**Batterie Puissance (PB)**\n\n"
            "- Très forte puissance instantanée\n"
            "- Répond aux pics de demande et au freinage\n"
            "- Protège la batterie Énergie des sollicitations brutales"
        )
with comp3:
    with st.popover("🔄 Convertisseur", use_container_width=True):
        st.markdown(
            "**Convertisseur**\n\n"
            "- Répartit dynamiquement la puissance entre les deux batteries\n"
            "- Possède ses propres limites de puissance à respecter"
        )
with comp4:
    with st.popover("🚗 Moteur et véhicule", use_container_width=True):
        st.markdown(
            "**Moteur et véhicule**\n\n"
            "- Le cycle de conduite impose à chaque instant une puissance demandée\n"
            "- En freinage, le moteur renvoie de l'énergie à récupérer"
        )


# KPI, libellés orientés domaine

n_strat, n_ia, n_pts = _kpis()
pts_txt = f"{n_pts:,}".replace(",", " ") if n_pts else "—"

st.markdown(
    f"""
    <div class="s2-kpi-grid">
      <div class="s2-kpi"><div class="n" style="color:{C_BLEU}">{n_strat}</div><div class="l">Stratégies EMS disponibles</div></div>
      <div class="s2-kpi"><div class="n" style="color:{C_VERT}">{n_ia}</div><div class="l">Modèles IA évalués</div></div>
      <div class="s2-kpi"><div class="n" style="color:{C_ORANGE}">{pts_txt}</div><div class="l">Instants de conduite simulés</div></div>
      <div class="s2-kpi"><div class="n" style="color:{C_VERT}">100 %</div><div class="l">Décisions expliquées</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Le défi : problème -> difficulté -> réponse

st.subheader("Le défi")

d1, d2, d3 = st.columns(3)
with d1:
    st.markdown(
        f'<div class="s2-card"><div class="t" style="color:{C_BLEU}">Le problème</div>'
        '<div class="d">Un véhicule électrique équipé de deux batteries complémentaires '
        'doit décider, à chaque instant, laquelle fournit la puissance demandée.</div></div>',
        unsafe_allow_html=True,
    )
with d2:
    st.markdown(
        f'<div class="s2-card"><div class="t" style="color:{C_ORANGE}">Pourquoi c\'est difficile</div>'
        '<div class="d">Les objectifs sont multiples et contradictoires : autonomie, durée de '
        'vie, rendement. Les contraintes physiques sont strictes et les cycles de conduite '
        'très variables.</div></div>',
        unsafe_allow_html=True,
    )
with d3:
    st.markdown(
        f'<div class="s2-card"><div class="t" style="color:{C_VERT}">La réponse de 2SMART</div>'
        '<div class="d">Simuler le système, comparer sept stratégies de gestion d\'énergie, '
        'et expliquer chaque décision — sous le contrôle d\'un filtre physique de sécurité.</div></div>',
        unsafe_allow_html=True,
    )

etapes = ["Cycle", "Prétraitement", "Modèle IA", "Filtre physique", "Simulation", "Explicabilité"]
pipe = '<div class="s2-pipe">'
for i, e in enumerate(etapes):
    pipe += f'<span class="s2-step">{e}</span>'
    if i < len(etapes) - 1:
        pipe += '<span class="s2-arrow">&#8594;</span>'
pipe += "</div>"
st.markdown(pipe, unsafe_allow_html=True)


# Que vais-je obtenir ?

st.subheader("Que vais-je obtenir ?")

benefices = [
    ("🧪 Simuler", "Observer la répartition de puissance sur un cycle de conduite complet.", C_BLEU),
    ("📊 Comparer", "Identifier la stratégie la plus performante selon le critère qui vous importe.", C_ORANGE),
    ("🧠 Expliquer", "Comprendre pourquoi une décision a été prise, à n'importe quel instant.", C_VERT),
    ("⚙️ Optimiser", "Trouver le meilleur compromis entre performance et préservation des batteries.", C_ARDOISE),
]
cols_ben = st.columns(4)
for col, (titre, desc, coul) in zip(cols_ben, benefices):
    with col:
        st.markdown(
            f'<div class="s2-card"><div class="t" style="color:{coul}">{titre}</div>'
            f'<div class="d">{desc}</div></div>',
            unsafe_allow_html=True,
        )


# Valeur scientifique

st.subheader("Pourquoi 2SMART ?")

with st.container(border=True):
    st.markdown(
        "- **IA hybride** : connaissances expertes de l'ontologie OntoHESS combinées à "
        "l'apprentissage, via une approche **neuro-symbolique**\n"
        "- **Respect des contraintes physiques** garanti par un filtre de sécurité en aval "
        "de chaque décision\n"
        "- **Explicabilité** des décisions, adaptée à la nature de chaque modèle\n"
        "- **Comparaison** de sept stratégies EMS sur des critères communs\n"
        "- **Validation par simulation** sur un cycle de conduite réel"
    )


# Comprendre la décision (équations repliées)

st.subheader("Comprendre la décision")

st.write(
    "À chaque instant, une stratégie choisit `alpha(t)` : la fraction de la puissance "
    "demandée confiée à la batterie de puissance. Tout le reste en découle."
)

with st.expander("Voir le modèle mathématique"):
    eq1, eq2 = st.columns(2)
    with eq1:
        st.latex(r"P_{PB} = \alpha \times P_{dem}")
    with eq2:
        st.latex(r"P_{EB} = (1 - \alpha) \times P_{dem}")
    st.caption(
        "alpha = 0 : toute la puissance vient de la batterie Énergie. "
        "alpha = 1 : toute la puissance vient de la batterie Puissance. "
        "La décision passe ensuite par le filtre physique de sécurité."
    )


# Que souhaitez-vous faire ?

st.subheader("Que souhaitez-vous faire ?")

mode1, mode2 = st.columns(2)
with mode1:
    st.markdown(
        f'<div class="s2-card"><div class="t" style="color:{C_BLEU}">Explorer une démonstration</div>'
        '<div class="d">Résultats déjà calculés, affichage instantané. Comparaison, analyse '
        'et explicabilité des sept stratégies.</div></div>',
        unsafe_allow_html=True,
    )
with mode2:
    st.markdown(
        f'<div class="s2-card"><div class="t" style="color:{C_VERT}">Lancer une simulation</div>'
        '<div class="d">Importer un nouveau cycle de conduite et simuler le système de bout '
        'en bout. Temps de calcul plus long.</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

acces1, acces2, acces3 = st.columns(3)
with acces1:
    if st.button("Explorer les résultats", type="primary", use_container_width=True):
        st.switch_page("vues/5_Comparaison_des_strategies.py")
with acces2:
    if st.button("Architecture des modèles", use_container_width=True):
        st.switch_page("vues/9_Architecture_des_modeles.py")
with acces3:
    if st.button("Préparation des données", use_container_width=True):
        st.switch_page("vues/2_Preparation_donnees.py")
