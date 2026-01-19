import streamlit as st
import pandas as pd
import os
import base64
import plotly.express as px
import plotly.graph_objects as go
from ift6758.client.serving_client import ServingClient
from ift6758.client.game_client import GameClient

# Informations sur les fonctionnalités BONUS ajoutées (Partie 7)
st.info(
    """
    #### Fonctionnalités Bonus (Partie 7)

    On a ajouté des visualisations dynamiques et intéractives (hover, zoom, etc) :
    1. **Position des tirs :** Affiche la localisation des tirs sur la glace, colorés par équipe, avec une taille proportionnelle à la probabilité de but (dangerosité du tir).
    Lors du survol de la souris sur un tir, une liste d'informations sur le tir est affichée.
    2. **Évolution des *expected goals (xG)* par équipe :** Graphique montrant le cumul des probabilités des buts attendus en fonction du temps.
    Indique quelle équipe domine le match au fil du temps.
    """
)

st.divider()

st.title("NHL Goal prediction")

# Garde un seul client ServingClient pour toute la session
if "serving_client" not in st.session_state:
    # Obtenir dynamiquement l'adresse IP du backend ("127.0.0.1" par défaut pour tester en local)
    serving_ip = os.environ.get("SERVING_IP", "127.0.0.1")
    st.session_state["serving_client"] = ServingClient(ip=serving_ip, port=5000)

serving_client = st.session_state["serving_client"]

# Garde un seul client GameClient pour toute la session
if "game_client" not in st.session_state:
    st.session_state["game_client"] = GameClient(serving_client=serving_client)
game_client = st.session_state["game_client"]

# Buffer persistant : stocke les événements par game_id pour éviter de les recharger à chaque refresh
if "events_buffers" not in st.session_state:
    st.session_state["events_buffers"] = {}

# Shortcuts
serving_client = st.session_state["serving_client"]
game_client = st.session_state["game_client"]

# Sidebar pour le chargement du modèle
with st.sidebar:
    st.header("Model Configuration")
    workspace = st.text_input("Workspace", value="IFT6758--2025-A03")
    model = st.text_input("Model", value="LogReg_Model_with_distance_and_angle")
    version = st.text_input("Version", value="v6")
    get_model = st.button("Get Model")

    if get_model:
        # Downloader le modèle depuis le registre
        with st.spinner("Downloading model..."):
            try:
                result = serving_client.download_registry_model(
                    entity=workspace,
                    artifact_name=model,
                    project="IFT6758.2025-A03",
                    version=version
                )

                # Mettre à jour les features à utiliser si fournies
                if 'features' in result and result['features'] is not None:
                    serving_client.features = result['features']
                    st.sidebar.success(f"Model features set: {serving_client.features}")

                st.success(f"Model downloaded successfully")
            except Exception as e:
                st.error(f"Failed to download model: {e}")

# Container qui permtet de pinger un jeu et obtenir les données + prédictions
with st.container():
    game_id = st.text_input("Game ID", value="2023020001")

    if 'previous_game_id' not in st.session_state:
        st.session_state['previous_game_id'] = game_id

    # Si l'utilisateur change de game_id, réinitialiser le buffer et les équipes
    if game_id != st.session_state['previous_game_id']:
        if game_id not in st.session_state["events_buffers"]:
            st.session_state["events_buffers"][game_id] = pd.DataFrame()
        st.session_state["home_team"] = None
        st.session_state["away_team"] = None
        st.session_state["new_events_df"] = pd.DataFrame()
        st.session_state['previous_game_id'] = game_id

    # Désactiver le bouton si aucun modèle n'est chargé
    ping_disabled = (serving_client.model is None)
    if ping_disabled:
        st.warning("Aucun modèle chargé — veuillez en charger un dans la barre latérale avant de continuer")
    ping_game = st.button("Ping Game", disabled=ping_disabled)

    if ping_game:
        with st.spinner("Fetching game data..."):

            # Process and predict
            new_events_df, home_team, away_team, current_score_home, current_score_away = game_client.process_and_predict(
                game_id)

            # Charger les équipes et le nouveau chunk dans le state
            st.session_state["home_team"] = home_team
            st.session_state["away_team"] = away_team

            if not new_events_df.empty:
                st.session_state["new_events_df"] = new_events_df

                # Mettre à jour le buffer des événements
                if game_id not in st.session_state["events_buffers"]:
                    st.session_state["events_buffers"][game_id] = pd.DataFrame()
                # Ajouter les nouveaux événements au buffer existant
                st.session_state["events_buffers"][game_id] = pd.concat(
                    [st.session_state["events_buffers"][game_id], new_events_df],
                    ignore_index=True)
            else:
                st.info("No new events — showing last known state.")

with st.container():
    if "events_buffers" in st.session_state and game_id in st.session_state["events_buffers"]:

        buffer = st.session_state["events_buffers"][game_id]

        if buffer.empty:
            st.stop()

        home_team = st.session_state["home_team"]
        away_team = st.session_state["away_team"]

        # Dernier état du match
        last = buffer.iloc[-1]

        goals = buffer[buffer["Event Type"] == "Goal"]
        # Scores actuels
        score_home = goals[goals["Team"].astype(str).str.strip().str.lower() ==
                           goals["Home Team"].astype(str).str.strip().str.lower()].shape[0]

        score_away = goals[goals["Team"].astype(str).str.strip().str.lower() !=
                           goals["Home Team"].astype(str).str.strip().str.lower()].shape[0]

        # Dernière période
        period = last["Period"]

        # Temps restant
        seconds_elapsed = last['Game Seconds']
        seconds_remaining = (20 * 60 * period) - seconds_elapsed
        minutes = seconds_remaining // 60
        seconds = seconds_remaining % 60
        clock_display = f"{int(minutes):02}:{int(seconds):02}"

        # Calculer les xG totaux par équipe
        home_xg = buffer[buffer["Team"].astype(str).str.strip().str.lower()
                         == buffer["Home Team"].astype(str).str.strip().str.lower()]["prediction"].sum()

        away_xg = buffer[buffer["Team"].astype(str).str.strip().str.lower() !=
                         buffer["Home Team"].astype(str).str.strip().str.lower()]["prediction"].sum()

        # Différence entre score réel et xG
        diff_home = score_home - home_xg
        diff_away = score_away - away_xg

        # Afficher le tableau de bord
        st.markdown(f"## Game {game_id}: **{home_team} vs {away_team}**")
        st.write(f"**Period {period} — {clock_display} left**")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label=f"{home_team} xG (actual)",
                value=f"{home_xg:.2f} ({score_home})",
                delta=f"{diff_home:.2f}"
            )
        with col2:
            st.metric(
                label=f"{away_team} xG (actual)",
                value=f"{away_xg:.2f} ({score_away})",
                delta=f"{diff_away:.2f}"
            )

with st.container():
    st.subheader("Données utilisées pour les prédictions")
    cols_to_show = ["Distance", "Angle", "Goal", "Empty Net", "prediction"]
    if "events_buffers" in st.session_state and game_id in st.session_state["events_buffers"]:
        buffer = st.session_state["events_buffers"][game_id]

    if "new_events_df" in st.session_state:
        # Si le nouveau chunk est vide, afficher le dernier chunk du buffer
        if st.session_state["new_events_df"].empty:
            # Afficher le dernier chunk du buffer
            if "events_buffers" in st.session_state and game_id in st.session_state["events_buffers"]:
                st.write(buffer[cols_to_show])
            else:
                st.write("Aucune donnée disponible pour le moment.")
        else:
            # Afficher le nouveau chunk
            st.write(buffer[cols_to_show])

with st.container():
    st.subheader("Visualisations")

    if "events_buffers" in st.session_state and game_id in st.session_state["events_buffers"]:
        df = st.session_state["events_buffers"][game_id]

        if not df.empty and "prediction" in df.columns:
            # Noms des équipes
            home_team = st.session_state.get("home_team", "Home")
            away_team = st.session_state.get("away_team", "Away")

            tab1, tab2 = st.tabs(["Position des tirs", "Évolution des buts attendus (xG)"])

            # --- Graphique des tirs ---
            with tab1:
                x_col = "X"
                y_col = "Y"

                if x_col in df.columns and y_col in df.columns:

                    df_rounded = df.copy()
                    df_rounded['prediction'] = df['prediction'].round(4)

                    fig_shot = px.scatter(
                        df_rounded,
                        x=x_col,
                        y=y_col,
                        color="Team",
                        size="prediction",  # Taille proportionnelle à la probabilité de but
                        size_max=15,
                        hover_data=["Event Type", "Shooter", "prediction", "Period", "Time"],
                        title=f"Position des tirs - {away_team} vs {home_team}",
                        color_discrete_sequence=["blue", "red"],
                        opacity=0.8
                    )

                    rink_image_path = "nhl_rink.png"

                    # Encodage
                    with open(rink_image_path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode()

                    # Ajouter l'image de fond
                    fig_shot.add_layout_image(
                        dict(
                            source='data:image/png;base64,' + encoded_image,
                            xref="x",
                            yref="y",
                            x=-100,
                            y=42.5,
                            sizex=200,
                            sizey=85,
                            sizing="contain",
                            opacity=1,
                            layer="below"
                        )
                    )

                    # Fixer les axes pour correspondre à la patinoire et éliminer la grille
                    fig_shot.update_layout(
                        xaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            visible=True,
                            range=[-100, 100]
                        ),
                        yaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            visible=True,
                            range=[-42.5, 42.5],
                            scaleanchor="x",
                            scaleratio=1  # Éviter la déformation de l'image
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        # Assurer que les points n'apparaissent pas hors de la zone définie par l'image
                        modebar_remove=['zoom', 'pan', 'select', 'lasso', 'autoscale', 'resetview']
                    )

                    st.plotly_chart(fig_shot, use_container_width=True)
                else:
                    st.warning(f"Impossible d'afficher la position des tirs.")

            # --- Évolution des xG ---
            with tab2:
                if "Game Seconds" in df.columns:
                    # Séparer les données par équipe
                    df_home = df[df['Team'].str.strip() == df['Home Team'].str.strip()].copy()
                    df_away = df[df['Team'].str.strip() != df['Home Team'].str.strip()].copy()

                    # Calcul de la somme cumulative des xG
                    df_home['Cumulative xG'] = df_home['prediction'].cumsum()
                    df_away['Cumulative xG'] = df_away['prediction'].cumsum()

                    fig = go.Figure()

                    hover_template = '<b>Team:</b> %{data.name}<br>' + \
                                     '<b>Game Seconds:</b> %{x:.2f}<br>' + \
                                     '<b>Cumulative xG:</b> %{y:.2f}<br>' + \
                                     '<extra></extra>'

                    # Ligne pour Home Team
                    fig.add_trace(go.Scatter(
                        x=df_home['Game Seconds'],
                        y=df_home['Cumulative xG'],
                        mode='lines+markers',
                        name=f"{home_team}",
                        line=dict(width=3, color='blue'),
                        hovertemplate=hover_template
                    ))

                    # Ligne pour Away Team
                    fig.add_trace(go.Scatter(
                        x=df_away['Game Seconds'],
                        y=df_away['Cumulative xG'],
                        mode='lines+markers',
                        name=f"{away_team}",
                        line=dict(width=3, color='red'),
                        hovertemplate=hover_template
                    ))

                    fig.update_layout(
                        title="Évolution des buts attendus (xG) au cours du match",
                        xaxis_title="Game seconds",
                        yaxis_title="Cumulative xG",
                        showlegend=True,
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Impossible d'afficher l'évolution temporelle des buts attendus (XG).")