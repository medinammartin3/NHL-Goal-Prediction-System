import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))     # /src

from src.features.Pbp_DataFrame import Pbp_to_DataFrame
from src.visualizations.visualizations import divide_N_zone, get_event_distance


def get_event_angle(event):

    # Coordonnées des cages
    left_goal = (-89, 0)
    right_goal = (89, 0)

    # Coordonnées du tir
    x_coord = event['X']
    y_coord = event['Y']

    # Zone depuis laquelle le tir a été effectué
    zone = event['Zone']
    
    # On ignore les tirs qui ont des coordonnées manquantes
    if pd.isna(x_coord) or pd.isna(y_coord):
        return None

    # Choisir quelle distance est la correcte
    if zone == 'O' or zone == 'NO':
        goal_x, goal_y = (right_goal if x_coord > 0 else left_goal)
    elif zone == 'D' or zone == 'ND':
        goal_x, goal_y = (left_goal if x_coord > 0 else right_goal)
    else:   # zone = None (donnée manquante)
        goal_x, goal_y = (right_goal if np.abs(x_coord - right_goal[0]) < np.abs(x_coord - left_goal[0]) else left_goal)
    
    dx = goal_x - x_coord
    dy = goal_y - y_coord

    angle = np.arctan2(np.abs(dy), np.abs(dx))
    angle = np.degrees(angle)

    return angle.round(1)

# Add a binary column if its a goal or not
def goal_or_not(df):
    df = df.dropna(subset=["Event Type"])
    df["Goal"] = (df["Event Type"] == "Goal").astype(int)
    return df

# Add a binary column if the net is empty or not
def get_net_situation(df):
    df['Empty Net'] = (df["Net"] == "Empty").astype(int)
    return df


def get_shots_angle_and_distance(seasons : list, rounded=False):
    # Récupérer les données
    df = get_multiple_season_df(seasons) 

    # Calculer la distance et angle
    df['Angle'] = df.apply(get_event_angle, axis=1)  
    df['Distance'] = df.apply(get_event_distance, axis=1)
    
    # Éliminer les tirs avec type inconnu (type de tir manquant)
    df = df[df['Type of Shot'] != "Unknown"] 

    # Éliminer les tirs sans distance ou sans angle(coordonnées manquantes)
    df = df.dropna(subset=['Angle']) 
    df = df.dropna(subset=['Distance'])
    
    if rounded:
        # Retourner aussi une version où les distances et angles ont été arrondies à l'entier le plus proche
        rounded_df = df.copy()
        rounded_df['Angle'] = rounded_df['Angle'].round().astype(int)
        rounded_df['Distance'] = rounded_df['Distance'].round().astype(int)
        return rounded_df
    else:
        return df


def get_shots_angle_and_distance_df(df, rounded=False):
    # Calculer la distance et angle
    df['Angle'] = df.apply(get_event_angle, axis=1)  
    df['Distance'] = df.apply(get_event_distance, axis=1)
    
    # Éliminer les tirs avec type inconnu (type de tir manquant)
    df = df[df['Type of Shot'] != "Unknown"] 

    # Éliminer les tirs sans distance ou sans angle(coordonnées manquantes)
    df = df.dropna(subset=['Angle']) 
    df = df.dropna(subset=['Distance'])
    
    if rounded:
        # Retourner aussi une version où les distances et angles ont été arrondies à l'entier le plus proche
        rounded_df = df.copy()
        rounded_df['Angle'] = rounded_df['Angle'].round().astype(int)
        rounded_df['Distance'] = rounded_df['Distance'].round().astype(int)
        return rounded_df
    else:
        return df

def get_multiple_season_df(seasons_list : list):
    data_path = '../../games_data'                   # Path vers les données
    
    games_dfs = []
    
    # Transformer en DataFrame les événements de chaque match pour chaque saison et
    # sauvegarder chaque DataFrame dans une liste
    for season in seasons_list:
        season_folder = os.path.join(data_path, season)  # Path des données de la saison
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
    
    # Nettoyer les valeurs pour éviter des problèmes de comparaison
    for col in season_df.select_dtypes(include="object"):
        season_df[col] = season_df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        
    # Ajouter une colonne qui indique si le tir a été affectué par l'équipe local
    season_df['Home Team Shot'] = season_df['Team'] == season_df['Home Team'].str.strip()
    
    # Formater la zone des événements en zone neutre
    season_df = divide_N_zone(season_df)
    
    return season_df


# Build the dataframe ask to train the first logisitic regression model
def build_feature_df(seasons : list, name : str):
    df = get_shots_angle_and_distance(seasons)
    df = goal_or_not(df)
    df = get_net_situation(df)
    print(df[["X", "Y"]])
    df = df[['Distance', 'Angle', 'Goal', 'Empty Net']]

    df.to_csv(f'../../games_data/{name}', index=False)

    return df

if __name__=="__main__":
    df_train = build_feature_df(["2016-2017", "2017-2018", "2018-2019", "2019-2020"], 'feature_dataset_1_train.csv')
    df_test = build_feature_df(['2020-2021'], 'feature_dataset_1_test.csv')