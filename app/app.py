import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Football Analytics System",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: #8ab26d; /* світло-оливковий */
    line-height: 1.2;
    margin-bottom: 6px;
}

.sub-header {
    text-align: center;
    font-size: 16px;
    color: #9ca3af;
    margin-bottom: 12px;
}

.header-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #A3B18A, transparent);
    margin: 10px 0 20px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

div[data-baseweb="tab-highlight"] {
    background-color: #8ab26d !important;
}

div[role="tablist"] > div {
    background-color: #8ab26d !important;
}

button[role="tab"][aria-selected="true"] p {
    color: #8ab26d !important;
}

button[role="tab"]:hover p {
    color: #8ab26d !important;
}

button:focus {
    outline: none !important;
    box-shadow: none !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

div[data-baseweb="select"] > div {
    border: 1px solid #2a2f36 !important;
}

div[data-baseweb="select"] > div:focus-within {
    border: 1px solid #8ab26d !important;
    box-shadow: 0 0 0 1px #8ab26d !important;
}

div[data-baseweb="select"]:hover > div {
    border: 1px solid #8ab26d !important;
}

div[role="listbox"] div[aria-selected="true"] {
    background-color: rgba(138, 178, 109, 0.25) !important;
}

div[role="listbox"] div:hover {
    background-color: rgba(138, 178, 109, 0.15) !important;
}

* {
    outline: none !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

a.anchor-link {
    display: none !important;
}

h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)


# HEADER
st.markdown(
    """
    <div class='main-header'>
        Система аналізу та прогнозування<br>
        футбольних матчів
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-header'>Аналіз • Візуалізація • Прогнозування</div>",
    unsafe_allow_html=True
)

st.markdown("<div class='header-divider'></div>", unsafe_allow_html=True)


# LOAD DATA
@st.cache_data
def load_data():
    data_clean = {
        "Premier League": pd.read_csv("../data/processed/premier_league_clean.csv"),
        "Bundesliga": pd.read_csv("../data/processed/bundesliga_clean.csv"),
        "League 1": pd.read_csv("../data/processed/league_1_clean.csv")
    }

    data_features = {
        "Premier League": pd.read_csv("../data/processed/premier_league_features.csv"),
        "Bundesliga": pd.read_csv("../data/processed/bundesliga_features.csv"),
        "League 1": pd.read_csv("../data/processed/league_1_features.csv")
    }

    return data_clean, data_features


data_clean, data_features = load_data()


# LOAD MODELS
@st.cache_resource
def load_models():
    return {
        "Premier League": joblib.load("../models/premier_league_ensemble.pkl"),
        "Bundesliga": joblib.load("../models/bundesliga_ensemble.pkl"),
        "League 1": joblib.load("../models/league_1_ensemble.pkl")
    }

models = load_models()


# TOP BAR
col1, col2 = st.columns([4, 1])

with col1:
    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

with col2:
    league = st.selectbox("Ліга", list(data_clean.keys()))

df_clean = data_clean[league]
df_features = data_features[league]

model = models[league]


# LOGO FUNCTION
def show_logo(team):
    path = f"assets/logos/{team.replace(' ', '_')}.png"
    if os.path.exists(path):
        st.image(path, width=120)
    else:
        st.write(team)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Аналіз команди",
    "Порівняння",
    "Прогноз",
    "Моделі",
    "Аналітика ліг"
])

# =========================
# TAB 1 — АНАЛІЗ КОМАНДИ
# =========================
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = df_clean

with tab1:

    st.header("Аналіз команди")

    col_sel1, col_sel2 = st.columns(2)

    with col_sel1:
        teams = sorted(df["HomeTeam"].unique())
        team = st.selectbox("Команда", teams)

    with col_sel2:
        seasons = sorted(df["Season"].unique())
        season = st.selectbox("Сезон", seasons)

    if league == "Premier League":
        df = pd.read_csv("../data/processed/premier_league_clean.csv")
    elif league == "Bundesliga":
        df = pd.read_csv("../data/processed/bundesliga_clean.csv")
    else:
        df = pd.read_csv("../data/processed/league_1_clean.csv")

    df_team = df[
        (
            (df["HomeTeam"] == team) |
            (df["AwayTeam"] == team)
        ) &
        (df["Season"] == season)
    ].copy()

    df_team = df_team.sort_values("Date").reset_index(drop=True)

    df_team["GoalsScored"] = np.where(
        df_team["HomeTeam"] == team,
        df_team["FTHG"],
        df_team["FTAG"]
    )

    df_team["GoalsConceded"] = np.where(
        df_team["HomeTeam"] == team,
        df_team["FTAG"],
        df_team["FTHG"]
    )

    df_team["Shots"] = np.where(
        df_team["HomeTeam"] == team,
        df_team["HS"],
        df_team["AS"]
    )

    df_team["ShotsOnTarget"] = np.where(
        df_team["HomeTeam"] == team,
        df_team["HST"],
        df_team["AST"]
    )

    df_team["Result"] = np.where(
        (
            ((df_team["HomeTeam"] == team) & (df_team["FTR"] == "H")) |
            ((df_team["AwayTeam"] == team) & (df_team["FTR"] == "A"))
        ),
        "Перемога",
        np.where(df_team["FTR"] == "D", "Нічия", "Поразка")
    )

    df_team["Points"] = np.where(
        df_team["Result"] == "Перемога",
        3,
        np.where(df_team["Result"] == "Нічия", 1, 0)
    )

    df_team["CumulativePoints"] = df_team["Points"].cumsum()

    win_rate = (df_team["Result"] == "Перемога").mean() * 100

    goals_scored = df_team["GoalsScored"].mean()

    goals_conceded = df_team["GoalsConceded"].mean()

    clean_sheets = (df_team["GoalsConceded"] == 0).sum()

    shots = df_team["Shots"].mean()

    shots_on_target = df_team["ShotsOnTarget"].mean()

    conversion_rate = (
        df_team["GoalsScored"].sum() /
        df_team["ShotsOnTarget"].sum()
    ) * 100 if df_team["ShotsOnTarget"].sum() > 0 else 0

    max_streak = 0
    current_streak = 0

    for result in df_team["Result"]:

        if result == "Перемога":
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    df_team["GoalDiff"] = (
        df_team["GoalsScored"] -
        df_team["GoalsConceded"]
    )

    best_match = df_team.loc[df_team["GoalDiff"].idxmax()]
    worst_match = df_team.loc[df_team["GoalDiff"].idxmin()]

    biggest_win = (
        f"{best_match['GoalsScored']}:"
        f"{best_match['GoalsConceded']}"
    )

    worst_loss = (
        f"{worst_match['GoalsScored']}:"
        f"{worst_match['GoalsConceded']}"
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        show_logo(team)

    with col2:
        st.subheader(f"{team} ({season})")
        st.caption(f"Матчів: {len(df_team)}")

    st.markdown("### Ключові показники")

    card_style = """
    padding:18px;
    border-radius:14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(138,178,109,0.15);
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    text-align:center;
    """

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div style='{card_style}'>
            <p>Відсоток перемог</p>
            <h2 style='color:#8ab26d'>{win_rate:.1f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div style='{card_style}'>
            <p>Голи за гру</p>
            <h2 style='color:#8ab26d'>{goals_scored:.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div style='{card_style}'>
            <p>Пропущені голи за гру</p>
            <h2 style='color:#8ab26d'>{goals_conceded:.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div style='{card_style}'>
            <p>Сухі матчі</p>
            <h2 style='color:#8ab26d'>{clean_sheets}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        st.markdown(
            f"""
            <div style='{card_style}'>
            <p>Удари</p>
            <h2 style='color:#8ab26d'>{shots:.1f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c6:
        st.markdown(
            f"""
            <div style='{card_style}'>
            <p>Удари в ствір</p>
            <h2 style='color:#8ab26d'>{shots_on_target:.1f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c7:
        st.markdown(
            f"""
            <div style='{card_style}'>
            <p>Реалізація ударів</p>
            <h2 style='color:#8ab26d'>{conversion_rate:.1f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c8:
        st.markdown(
            f"""
            <div style='{card_style}'>
            <p>Найдовша серія перемог</p>
            <h2 style='color:#8ab26d'>{max_streak}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

    c9, c10 = st.columns(2)

    with c9:
        st.markdown(
            f"""
            <div style='{card_style}'>
            <p>Найбільша перемога</p>
            <h2 style='color:#8ab26d'>{biggest_win}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c10:
        st.markdown(
            f"""
            <div style='{card_style}'>
            <p>Найбільша поразка</p>
            <h2 style='color:#8ab26d'>{worst_loss}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Аналітика та динаміка")

    col_left, col_right = st.columns(2)

    with col_left:

        results_counts = df_team["Result"].value_counts()

        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=results_counts.index,
                    values=results_counts.values,
                    hole=0.6,
                    marker=dict(
                        colors=[
                            "#8ab26d",
                            "#d4b96e",
                            "#a14d4d"
                        ]
                    )
                )
            ]
        )

        fig_pie.update_layout(
            title="Результати матчів",
            template="plotly_dark",
            height=400,
            margin=dict(l=10, r=10, t=50, b=10)
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with col_right:

        fig_points = go.Figure()

        fig_points.add_trace(
            go.Scatter(
                x=list(range(1, len(df_team) + 1)),
                y=df_team["CumulativePoints"],
                mode="lines+markers",
                line=dict(
                    color="#8ab26d",
                    width=4
                ),
                fill="tozeroy"
            )
        )

        fig_points.update_layout(
            title="Кумулятивні очки",
            template="plotly_dark",
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Тур",
            yaxis_title="Очки",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_points,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.markdown("### Динаміка позиції в таблиці")

    season_df = df[df["Season"] == season].copy()

    season_df = season_df.sort_values("Date").reset_index(drop=True)

    teams_all = sorted(season_df["HomeTeam"].unique())

    table_points = {
        t: 0 for t in teams_all
    }

    positions = []
    match_numbers = []

    team_match_number = 0

    for _, match in season_df.iterrows():

        home = match["HomeTeam"]
        away = match["AwayTeam"]
        result = match["FTR"]

        # =========================
        # UPDATE POINTS
        # =========================
        if result == "H":

            table_points[home] += 3

        elif result == "A":

            table_points[away] += 3

        else:

            table_points[home] += 1
            table_points[away] += 1

        if home == team or away == team:

            team_match_number += 1

            sorted_table = sorted(
                table_points.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for pos, (t, pts) in enumerate(sorted_table, start=1):

                if t == team:
                    positions.append(pos)
                    match_numbers.append(team_match_number)
                    break

    fig_position = go.Figure()

    fig_position.add_trace(
        go.Scatter(
            x=match_numbers,
            y=positions,
            mode="lines+markers",
            line=dict(
                color="#8ab26d",
                width=4
            ),
            marker=dict(
                size=7
            )
        )
    )

    fig_position.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Матч",
        yaxis_title="Позиція",
        hovermode="x unified",
        yaxis=dict(
            autorange="reversed",
            tickmode="linear",
            dtick=1
        )
    )

    fig_position.add_hrect(
        y0=1,
        y1=4,
        fillcolor="green",
        opacity=0.08,
        line_width=0
    )

    fig_position.add_hrect(
        y0=len(teams_all) - 2,
        y1=len(teams_all),
        fillcolor="red",
        opacity=0.08,
        line_width=0
    )

    st.plotly_chart(
        fig_position,
        use_container_width=True,
        config={"displayModeBar": False}
    )

import streamlit.components.v1 as components

df = df_clean

with tab2:

    st.header("Порівняння команд")

    teams = sorted(df["HomeTeam"].unique())
    seasons = sorted(df["Season"].unique())

    col1, col2 = st.columns(2)

    with col1:

        team1 = st.selectbox(
            "Команда",
            teams,
            key="t1"
        )

        season1 = st.selectbox(
            "Сезон",
            seasons,
            key="s1"
        )

        c1, c2, c3 = st.columns([1, 1, 1])

        with c2:
            show_logo(team1)

    with col2:

        team2 = st.selectbox(
            "Команда",
            teams,
            key="t2"
        )

        season2 = st.selectbox(
            "Сезон",
            seasons,
            key="s2"
        )

        c1, c2, c3 = st.columns([1, 1, 1])

        with c2:
            show_logo(team2)

    def get_stats(team, season):

        df_team = df[
            (
                (df["HomeTeam"] == team) |
                (df["AwayTeam"] == team)
            ) &
            (df["Season"] == season)
        ].copy()

        df_team = (
            df_team
            .sort_values("Date")
            .reset_index(drop=True)
        )

        goals = np.where(
            df_team["HomeTeam"] == team,
            df_team["FTHG"],
            df_team["FTAG"]
        )

        conceded = np.where(
            df_team["HomeTeam"] == team,
            df_team["FTAG"],
            df_team["FTHG"]
        )

        shots = np.where(
            df_team["HomeTeam"] == team,
            df_team["HS"],
            df_team["AS"]
        )

        shots_on_target = np.where(
            df_team["HomeTeam"] == team,
            df_team["HST"],
            df_team["AST"]
        )

        results = np.where(
            (
                (
                    (df_team["HomeTeam"] == team) &
                    (df_team["FTR"] == "H")
                ) |
                (
                    (df_team["AwayTeam"] == team) &
                    (df_team["FTR"] == "A")
                )
            ),
            "W",
            np.where(
                df_team["FTR"] == "D",
                "D",
                "L"
            )
        )

        points = np.where(
            results == "W",
            3,
            np.where(results == "D", 1, 0)
        )

        cumulative_points = np.cumsum(points)

        win_rate = (
            (results == "W").mean()
        ) * 100

        clean_sheets = (
            conceded == 0
        ).sum()

        conversion_rate = (
            goals.sum() /
            shots_on_target.sum()
        ) * 100 if shots_on_target.sum() > 0 else 0

        stats = {
            "Відсоток перемог": win_rate,
            "Голи за гру": np.mean(goals),
            "Пропущені голи": np.mean(conceded),
            "Сухі матчі": clean_sheets,
            "Удари": np.mean(shots),
            "Удари в ствір": np.mean(shots_on_target),
            "Реалізація ударів": conversion_rate
        }

        return (
            stats,
            cumulative_points,
            df_team
        )

    stats1, cumulative1, df_team1 = get_stats(
        team1,
        season1
    )

    stats2, cumulative2, df_team2 = get_stats(
        team2,
        season2
    )

    st.markdown("## Порівняння показників")

    metrics = list(stats1.keys())

    for m in metrics:

        v1 = stats1[m]
        v2 = stats2[m]

        max_val = max(v1, v2)

        if max_val == 0:
            max_val = 1

        p1 = (v1 / max_val) * 100
        p2 = (v2 / max_val) * 100

        color1 = "#8ab26d" if v1 >= v2 else "#556070"
        color2 = "#8ab26d" if v2 >= v1 else "#556070"

        bg_color = "#1f2937"

        col_left, col_center, col_right = st.columns([5, 2, 5])

        with col_left:

            components.html(
                f"""
                <div style="
                    width:100%;
                    font-family:sans-serif;
                    text-align:center;
                ">

                    <!-- VALUE -->
                    <div style="
                        margin-bottom:8px;
                        font-size:18px;
                        font-weight:700;
                        color:#2d3748;
                    ">
                        {v1:.2f}
                    </div>

                    <!-- BAR -->
                    <div style="
                        width:100%;
                        height:18px;
                        background:{bg_color};
                        border-radius:999px;
                        overflow:hidden;
                    ">

                        <div style="
                            margin-left:auto;
                            width:{p1}%;
                            height:100%;
                            background:{color1};
                            border-radius:999px;
                        ">
                        </div>

                    </div>

                </div>
                """,
                height=58
            )

        with col_center:

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    margin-top:18px;
                    font-size:18px;
                    font-weight:700;
                ">
                    {m}
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_right:

            components.html(
                f"""
                <div style="
                    width:100%;
                    font-family:sans-serif;
                    text-align:center;
                ">

                    <!-- VALUE -->
                    <div style="
                        margin-bottom:8px;
                        font-size:18px;
                        font-weight:700;
                        color:#2d3748;
                    ">
                        {v2:.2f}
                    </div>

                    <!-- BAR -->
                    <div style="
                        width:100%;
                        height:18px;
                        background:{bg_color};
                        border-radius:999px;
                        overflow:hidden;
                    ">

                        <div style="
                            width:{p2}%;
                            height:100%;
                            background:{color2};
                            border-radius:999px;
                        ">
                        </div>

                    </div>

                </div>
                """,
                height=58
            )

        st.markdown(
            "<div style='margin-bottom:4px'></div>",
            unsafe_allow_html=True
        )

    st.markdown("## Динаміка набору очок")

    fig_points = go.Figure()

    fig_points.add_trace(
        go.Scatter(
            x=list(range(1, len(cumulative1) + 1)),
            y=cumulative1,
            mode="lines+markers",
            name=f"{team1} ({season1})",
            line=dict(
                color="#8ab26d",
                width=4
            )
        )
    )

    fig_points.add_trace(
        go.Scatter(
            x=list(range(1, len(cumulative2) + 1)),
            y=cumulative2,
            mode="lines+markers",
            name=f"{team2} ({season2})",
            line=dict(
                color="#556070",
                width=4
            )
        )
    )

    fig_points.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Матч",
        yaxis_title="Очки",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_points,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    def calculate_positions(team, season):

        season_df = df[
            df["Season"] == season
        ].copy()

        season_df = (
            season_df
            .sort_values("Date")
            .reset_index(drop=True)
        )

        teams_all = sorted(
            season_df["HomeTeam"].unique()
        )

        table_points = {
            t: 0 for t in teams_all
        }

        positions = []
        match_numbers = []

        team_match_number = 0

        for _, match in season_df.iterrows():

            home = match["HomeTeam"]
            away = match["AwayTeam"]
            result = match["FTR"]

            if result == "H":

                table_points[home] += 3

            elif result == "A":

                table_points[away] += 3

            else:

                table_points[home] += 1
                table_points[away] += 1

            if home == team or away == team:

                team_match_number += 1

                sorted_table = sorted(
                    table_points.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                for pos, (t, pts) in enumerate(
                    sorted_table,
                    start=1
                ):

                    if t == team:

                        positions.append(pos)
                        match_numbers.append(
                            team_match_number
                        )
                        break

        return positions, match_numbers, len(teams_all)

    positions1, matches1, total_teams1 = (
        calculate_positions(team1, season1)
    )

    positions2, matches2, total_teams2 = (
        calculate_positions(team2, season2)
    )

    st.markdown("## Динаміка позицій у таблиці")

    fig_position = go.Figure()

    fig_position.add_trace(
        go.Scatter(
            x=matches1,
            y=positions1,
            mode="lines+markers",
            name=f"{team1} ({season1})",
            line=dict(
                color="#8ab26d",
                width=4
            )
        )
    )

    fig_position.add_trace(
        go.Scatter(
            x=matches2,
            y=positions2,
            mode="lines+markers",
            name=f"{team2} ({season2})",
            line=dict(
                color="#556070",
                width=4
            )
        )
    )

    fig_position.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Матч",
        yaxis_title="Позиція",
        hovermode="x unified",
        yaxis=dict(
            autorange="reversed",
            tickmode="linear",
            dtick=1
        )
    )

    fig_position.add_hrect(
        y0=1,
        y1=4,
        fillcolor="green",
        opacity=0.08,
        line_width=0
    )

    fig_position.add_hrect(
        y0=max(total_teams1, total_teams2) - 2,
        y1=max(total_teams1, total_teams2),
        fillcolor="red",
        opacity=0.08,
        line_width=0
    )

    st.plotly_chart(
        fig_position,
        use_container_width=True,
        config={"displayModeBar": False}
    )

# =========================
# TAB 3
# # =========================
df = df_features

with tab3:
    st.header("Прогноз матчу")

    teams = sorted(df["HomeTeam"].unique())

    def show_logo(team):
        path = f"assets/logos/{team.replace(' ', '_')}.png"
        if os.path.exists(path):
            st.image(path, width=160)
        else:
            st.write(team)

    st.markdown("""
    <style>

    /* BUTTON */
    div.stButton {
        margin-top: -40px;
        margin-bottom: -15px;
        text-align: center;
    }

    div.stButton > button {
        font-size: 18px;
        padding: 12px 20px;
        border-radius: 10px;
        border: 1.5px solid #8ab26d;
        color: #8ab26d;
        background-color: transparent;
    }

    div.stButton > button:hover {
        background-color: #8ab26d;
        color: white;
    }

    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([2, 3, 1, 3, 2])

    with c2:
        st.markdown("<div style='margin-top:25px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center; margin-bottom:-12px;'><b>Домашня команда</b></p>",
            unsafe_allow_html=True
        )
        home_team = st.selectbox("", teams, key="home_team")

    with c4:
        st.markdown("<div style='margin-top:25px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center; margin-bottom:-12px;'><b>Гостьова команда</b></p>",
            unsafe_allow_html=True
        )
        away_team = st.selectbox("", teams, key="away_team")

    with c3:
        st.markdown(
            "<h2 style='text-align:center; margin-top:55px; margin-left:12px;'>VS</h2>",
            unsafe_allow_html=True
        )

    with c1:
        show_logo(home_team)

    with c5:
        show_logo(away_team)

    b1, b2, b3 = st.columns([3, 2, 3])
    with b2:
        predict = st.button("СПРОГНОЗУВАТИ", use_container_width=True)

    st.markdown("---")

    if predict:

        model_bundle = model
        models_dict = model_bundle["models"]
        feature_cols = model_bundle["features"]

        base = df.tail(1).copy()

        home_df = df[df["HomeTeam"] == home_team].tail(5)
        away_df = df[df["AwayTeam"] == away_team].tail(5)

        base["Last5HomeOver2.5Perc"] = home_df["Last5HomeOver2.5Perc"].mean()
        base["Last5AwayOver2.5Perc"] = away_df["Last5AwayOver2.5Perc"].mean()

        base["AvgLast5HomeGoalsScored"] = home_df["AvgLast5HomeGoalsScored"].mean()
        base["AvgLast5HomeGoalsConceded"] = home_df["AvgLast5HomeGoalsConceded"].mean()

        base["AvgLast5AwayGoalsScored"] = away_df["AvgLast5AwayGoalsScored"].mean()
        base["AvgLast5AwayGoalsConceded"] = away_df["AvgLast5AwayGoalsConceded"].mean()

        X_input = base[feature_cols].values

        probs_list = []
        for m in models_dict.values():
            probs_list.append(m.predict_proba(X_input))

        avg_probs = np.mean(probs_list, axis=0)
        pred = np.argmax(avg_probs, axis=1)[0]
        probs = avg_probs[0]

        st.markdown("<div style='margin-top:-15px'></div>", unsafe_allow_html=True)

        r1, r2, r3 = st.columns([3,3,3])

        with r1:
            st.markdown("<p style='text-align:center'>Господарі</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align:center; color:#8ab26d'>{probs[2]*100:.1f}%</h2>", unsafe_allow_html=True)

        with r2:
            st.markdown("<p style='text-align:center'>Нічия</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align:center; color:#8ab26d'>{probs[1]*100:.1f}%</h2>", unsafe_allow_html=True)

        with r3:
            st.markdown("<p style='text-align:center'>Гості</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align:center; color:#8ab26d'>{probs[0]*100:.1f}%</h2>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        w1, w2, w3 = st.columns([3,3,3])

        with w2:
            if pred == 2:
                st.success("Перемога господарів")
            elif pred == 1:
                st.warning("Нічия")
            else:
                st.error("Перемога гостей")


# =========================
# TAB 4 — MODELS
# =========================

with tab4:

    st.header("Моделі машинного навчання")

    st.write("""
    Для прогнозування результатів футбольних матчів було використано
    декілька алгоритмів машинного навчання.
    Моделі аналізують статистичні показники команд,
    форму за останні матчі, атакувальні та захисні характеристики.
    """)

    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Random Forest", "63%")
        st.caption("Ансамбль дерев рішень")

    with c2:
        st.metric("XGBoost", "64%")
        st.caption("Градієнтний бустинг")

    with c3:
        st.metric("Logistic Regression", "61%")
        st.caption("Лінійна класифікація")

    with c4:
        st.metric("KNN", "58%")
        st.caption("Найближчі сусіди")

    with c5:
        st.metric("SVM", "61%")
        st.caption("Метод опорних векторів")

    st.markdown("---")

    st.success("""
    Найкращі результати показала модель XGBoost,
    яка забезпечила найвищу точність прогнозування.
    """)

    st.markdown("---")

    st.subheader("Матриці помилок")

    st.caption("""
    Матриці помилок демонструють,
    наскільки точно моделі класифікують
    результати футбольних матчів.
    """)

    league_key = league.lower().replace(" ", "_")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Random Forest")

        st.image(
            f"../plots/{league_key}_RandomForest_confusion_matrix.png",
            use_container_width=True
        )

    with col2:
        st.markdown("#### XGBoost")

        st.image(
            f"../plots/{league_key}_XGBoost_confusion_matrix.png",
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Важливість ознак")

    st.write("""
    Аналіз важливості ознак дозволяє визначити,
    які статистичні характеристики найбільше
    впливають на результат прогнозування.
    """)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Random Forest")

        st.image(
            f"../plots/{league_key}_rf_importance.png",
            use_container_width=True
        )

    with col2:

        st.markdown("#### XGBoost")

        st.image(
            f"../plots/{league_key}_xgb_importance.png",
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Висновки")

    st.markdown("""
    Ансамблеві моделі машинного навчання показали
    найкращу ефективність у задачі прогнозування
    результатів футбольних матчів.

    Найскладнішим для прогнозування результатом
    є нічия, що пов’язано з високою варіативністю
    футбольних подій.
    """)


# =========================
# TAB 5
# =========================

premier_df = pd.read_csv(
    "../data/processed/premier_league_clean.csv"
)

bundesliga_df = pd.read_csv(
    "../data/processed/bundesliga_clean.csv"
)

league1_df = pd.read_csv(
    "../data/processed/league_1_clean.csv"
)

with tab5:

    st.header("Аналітика ліг")

    st.markdown("""
    У вкладці представлено порівняльний аналіз футбольних ліг
    на основі статистичних показників матчів.

    Аналіз дозволяє оцінити результативність чемпіонатів,
    стиль гри команд та частоту результативних матчів.
    """)

    leagues = {

        "Premier League": premier_df,
        "Bundesliga": bundesliga_df,
        "League 1": league1_df

    }

    league_stats = []

    for league_name, league_df in leagues.items():

        avg_goals = np.mean(
            league_df["FTHG"] + league_df["FTAG"]
        )

        over25 = np.mean(
            (league_df["FTHG"] + league_df["FTAG"]) > 2
        ) * 100

        avg_shots = np.mean(
            league_df["HS"] + league_df["AS"]
        )

        avg_sot = np.mean(
            league_df["HST"] + league_df["AST"]
        )

        home_wins = np.mean(
            league_df["FTR"] == "H"
        ) * 100

        draws = np.mean(
            league_df["FTR"] == "D"
        ) * 100

        away_wins = np.mean(
            league_df["FTR"] == "A"
        ) * 100

        league_stats.append({

            "Ліга": league_name,

            "Сер. голів":
                avg_goals,

            "Over 2.5":
                over25,

            "Сер. ударів":
                avg_shots,

            "У ствір":
                avg_sot,

            "Home Win":
                home_wins,

            "Draw":
                draws,

            "Away Win":
                away_wins
        })

    stats_df = pd.DataFrame(league_stats)

    st.markdown("## Основні показники")

    c1, c2, c3 = st.columns(3)

    card_cols = [c1, c2, c3]

    for col, (_, row) in zip(card_cols, stats_df.iterrows()):

        league_df = leagues[row["Ліга"]]

        clean_sheets = np.mean(
            (
                    (league_df["FTHG"] == 0) |
                    (league_df["FTAG"] == 0)
            )
        ) * 100

        with col:
            with st.container(border=True):
                # =========================
                # TITLE
                # =========================
                st.markdown(
                    f"""
                    <h2 style="
                        text-align:center;
                        color:#8ab26d;
                        margin-top:5px;
                        margin-bottom:30px;
                        font-weight:700;
                    ">
                        {row['Ліга']}
                    </h2>
                    """,
                    unsafe_allow_html=True
                )

                m1, m2 = st.columns(2)

                with m1:
                    st.metric(
                        "Голів",
                        f"{row['Сер. голів']:.2f}"
                    )

                with m2:
                    st.metric(
                        "Сухі матчі",
                        f"{clean_sheets:.1f}%"
                    )

                m3, m4 = st.columns(2)

                with m3:
                    st.metric(
                        "Удари",
                        f"{row['Сер. ударів']:.1f}"
                    )

                with m4:
                    st.metric(
                        "У ствір",
                        f"{row['У ствір']:.1f}"
                    )

    g1, g2 = st.columns(2)

    with g1:

        st.markdown("### Середня результативність")

        fig_goals = go.Figure()

        fig_goals.add_trace(
            go.Bar(

                x=stats_df["Ліга"],
                y=stats_df["Сер. голів"],

                marker=dict(
                    color="#8ab26d"
                ),

                text=[
                    f"{x:.2f}"
                    for x in stats_df["Сер. голів"]
                ],

                textposition="outside"
            )
        )

        fig_goals.update_layout(

            template="plotly_dark",

            height=420,

            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),

            yaxis_title="Голи",

            showlegend=False
        )

        st.plotly_chart(
            fig_goals,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with g2:

        st.markdown("### Частота матчів Over 2.5")

        fig_over = go.Figure()

        fig_over.add_trace(
            go.Bar(

                x=stats_df["Ліга"],
                y=stats_df["Over 2.5"],

                marker=dict(
                    color="#5b6678"
                ),

                text=[
                    f"{x:.1f}%"
                    for x in stats_df["Over 2.5"]
                ],

                textposition="outside"
            )
        )

        fig_over.update_layout(

            template="plotly_dark",

            height=420,

            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),

            yaxis_title="% матчів",

            showlegend=False
        )

        st.plotly_chart(
            fig_over,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.markdown("## Розподіл результатів матчів")

    fig_results = go.Figure()

    fig_results.add_trace(
        go.Bar(

            name="Перемоги господарів",

            x=stats_df["Ліга"],
            y=stats_df["Home Win"],

            marker_color="#8ab26d"
        )
    )

    fig_results.add_trace(
        go.Bar(

            name="Нічиї",

            x=stats_df["Ліга"],
            y=stats_df["Draw"],

            marker_color="#5b6678"
        )
    )

    fig_results.add_trace(
        go.Bar(

            name="Перемоги гостей",

            x=stats_df["Ліга"],
            y=stats_df["Away Win"],

            marker_color="#111827"
        )
    )

    fig_results.update_layout(

        barmode="stack",

        template="plotly_dark",

        height=500,

        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),

        yaxis_title="% матчів"
    )

    st.plotly_chart(
        fig_results,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.markdown("## Висновок")

    st.markdown("""
    Проведений аналіз показав,
    що футбольні ліги мають різні статистичні особливості
    та рівень результативності.

    Найбільш результативні чемпіонати
    характеризуються більшою кількістю голів,
    ударів та матчів із тоталом більше 2.5 голів.

    Отримані результати дозволяють
    оцінити особливості стилю гри різних ліг
    та можуть бути використані
    для подальшого аналізу футбольних матчів.
    """)


# FOOTER
st.markdown("---")
st.markdown("Дипломна робота • Football Analytics System")