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
    .s2-hero-sub{font-size:1.15rem;color:#94A3B8;margin:.5rem 0 .3rem;font-weight:600}
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
    .s2-bar{color:#94A3B8;font-size:.9rem}
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


# Hero

hero_g, hero_d = st.columns([1.25, 1])
with hero_g:
    st.markdown(
        """
        <div class="s2-hero-title">2SMART</div>
        <div class="s2-hero-sub">Plateforme d'IA semi-supervisée pour systèmes hybrides de stockage</div>
        <div class="s2-hero-tags">Expliquer &nbsp;·&nbsp; Simuler &nbsp;·&nbsp; Comparer &nbsp;·&nbsp; Optimiser</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Nouvelle simulation", type="primary", use_container_width=True):
            st.switch_page("vues/8_Simulation_cycle_personnalise.py")
    with b2:
        if st.button("Voir les modèles", use_container_width=True):
            st.switch_page("vues/9_Architecture_des_modeles.py")

with hero_d:
    st.plotly_chart(_schema_hess(), use_container_width=True)


# KPI

n_strat, n_ia, n_pts = _kpis()
pts_txt = f"{n_pts:,}".replace(",", " ") if n_pts else "—"

st.markdown(
    f"""
    <div class="s2-kpi-grid">
      <div class="s2-kpi"><div class="n" style="color:{C_BLEU}">{n_strat}</div><div class="l">Stratégies EMS</div></div>
      <div class="s2-kpi"><div class="n" style="color:{C_VERT}">{n_ia}</div><div class="l">Modèles IA</div></div>
      <div class="s2-kpi"><div class="n" style="color:{C_ORANGE}">{pts_txt}</div><div class="l">Points simulés</div></div>
      <div class="s2-kpi"><div class="n" style="color:{C_VERT}">100 %</div><div class="l">Décisions explicables</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Pipeline

st.subheader("Chaîne de traitement")

etapes = ["Cycle", "Prétraitement", "Modèle IA", "Filtre physique", "Simulation", "Explicabilité"]
pipe = '<div class="s2-pipe">'
for i, e in enumerate(etapes):
    pipe += f'<span class="s2-step">{e}</span>'
    if i < len(etapes) - 1:
        pipe += '<span class="s2-arrow">&#8594;</span>'
pipe += "</div>"
st.markdown(pipe, unsafe_allow_html=True)


# Objectifs

st.subheader("Ce que fait la plateforme")

objectifs = [
    ("Simuler", "Préparer un cycle de conduite et rejouer le comportement du HESS (SOC, puissances EB/PB).", C_BLEU),
    ("Expliquer", "Justifier chaque décision de répartition en langage physique, modèle par modèle.", C_VERT),
    ("Comparer", "Classer sept stratégies EMS sur des critères communs de sécurité, coût et équilibre.", C_ORANGE),
    ("Optimiser", "Coupler règles expertes et apprentissage pour une décision sûre et interprétable.", C_ARDOISE),
]
cols_obj = st.columns(4)
for col, (titre, desc, coul) in zip(cols_obj, objectifs):
    with col:
        st.markdown(
            f'<div class="s2-card"><div class="t" style="color:{coul}">{titre}</div>'
            f'<div class="d">{desc}</div></div>',
            unsafe_allow_html=True,
        )


# Variable de décision

st.subheader("Variable de décision")

st.write(
    "À chaque instant, une stratégie choisit `alpha(t)` : la fraction de la puissance "
    "demandée confiée à la batterie de puissance. La décision passe ensuite par un filtre "
    "physique de sécurité."
)
eq1, eq2 = st.columns(2)
with eq1:
    st.latex(r"P_{PB} = \alpha \times P_{dem}")
with eq2:
    st.latex(r"P_{EB} = (1 - \alpha) \times P_{dem}")


# Deux modes

st.subheader("Deux modes d'utilisation")

mode1, mode2 = st.columns(2)
with mode1:
    st.markdown(
        f'<div class="s2-card"><div class="t" style="color:{C_BLEU}">Mode démonstration</div>'
        '<div class="d">Résultats précalculés, chargement instantané. Comparaison, analyse '
        'et explicabilité des sept stratégies. Idéal pour une soutenance.</div></div>',
        unsafe_allow_html=True,
    )
with mode2:
    st.markdown(
        f'<div class="s2-card"><div class="t" style="color:{C_VERT}">Mode simulation</div>'
        '<div class="d">Import d\'un nouveau cycle et simulation complète. Temps de calcul '
        'plus long, pour analyser un cycle inédit.</div></div>',
        unsafe_allow_html=True,
    )

st.write("")


# Accès

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
