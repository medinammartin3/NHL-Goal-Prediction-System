import pandas as pd
import os
from Pbp_DataFrame import Pbp_to_DataFrame

def get_season_df(season):
    data_path = '../../games_data'  # Path vers les données
    season_folder = os.path.join(data_path, season)  # Path des données de la saison

    games_dfs = []

    for game in os.listdir(season_folder):

        game_id = os.path.splitext(game)[0]
        game_df = Pbp_to_DataFrame().build_game_DataFrame(game_id)

        # Ne pas prendre en compte les matchs sans événements (données manquantes)
        if not game_df.empty:
            # Éliminer les colonnes sans valeurs (all NaN)
            game_df = game_df.dropna(axis=1, how='all')
            games_dfs.append(game_df)

    # Concaténer les DataFrames de chaque match pour former le DataFrame de la saison
    season_df = pd.concat(games_dfs, ignore_index=True)

    return season_df

def get_regular_and_playoff_games_ids(df):
    games_dict = {
        "regular": df.loc[df["Game Type"] == "Regular", "Game ID"].unique().tolist(),
        "playoffs": df.loc[df["Game Type"] == "Playoffs", "Game ID"].unique().tolist()
    }

    return games_dict

def split_regular_and_playoffs(folder, csv_name):

    season_df = get_season_df('2020-2021')
    num_games = season_df["Game ID"].nunique()
    print("Total games: " + str(num_games))

    games_dict = get_regular_and_playoff_games_ids(season_df)

    regular_games = games_dict["regular"]
    playoff_games = games_dict["playoffs"]
    print("Regular season games: " + str(len(regular_games)))
    print("Playoffs season games: " + str(len(playoff_games)))

    csv_path = os.path.join(folder, csv_name)
    test_csv_full = pd.read_csv(csv_path)

    # Séparer en saison régulière et playoffs
    regular_df = test_csv_full[test_csv_full["Game ID"].astype(str).isin(regular_games)].reset_index(drop=True)
    playoffs_df = test_csv_full[test_csv_full["Game ID"].astype(str).isin(playoff_games)].reset_index(drop=True)

    # new CSV path
    regular_path = os.path.join(folder, "feature_dataset_2_test_regular.csv")
    playoff_path = os.path.join(folder, "feature_dataset_2_test_playoff.csv")

    # Sauvegarde
    regular_df.to_csv(regular_path, index=False)
    playoffs_df.to_csv(playoff_path, index=False)

    print("Files saved successfully")


split_regular_and_playoffs("../../games_data", "feature_dataset_2_test.csv")
