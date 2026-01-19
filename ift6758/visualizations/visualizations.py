import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("../"))
from features.Pbp_DataFrame import Pbp_to_DataFrame


"""
Cette fonction prend en entrée une saison au choix (ex. '2023-2024') et
retourne un DataFrame contenant toutes les informations de chaque événement
de tous les matchs de cette saison.
"""
def get_season_df(season):
    data_path = '../../games_data'                   # Path vers les données
    season_folder = os.path.join(data_path, season)  # Path des données de la saison
    
    games_dfs = []
    
    # Transformer en DataFrame les événements de chaque match et
    # sauvegarder chaque DataFrame dans une liste
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
    
    
    

"""
Cette fonction prend en argument un DataFrame de saison et retourne un DataFrame
indiquant le nombre de tirs pour chaque type de tir.
"""
def get_shots_frequency_by_type(df):

    # Compter le nombre de tirs et buts (colonnes) pour chaque type (lignes)
    type_frequency_df = pd.crosstab(df['Type of Shot'], df['Event Type'])

    # Supprimer la ligne "Unknown" (tirs pour lesquels son type était manquant)
    type_frequency_df = type_frequency_df.drop(index="Unknown", errors="ignore")

    return type_frequency_df
    
    
    
    
"""
Cette fonction prend en entrée un DataFrame d'événements et modifie la zone
des tirs effectués dans la zone neutre (N) pour indiquer s'il a été effectué
dans la partie offensive (NO) ou défensive (ND) de la zone neutre.

Ceci nous aidera à identifier correctement de quel côté du terrain et 
vers quelle direction chaque tir à été effectué.
"""
def divide_N_zone(df):
    
    home_team_D_side_exists = 'Home Team D Side' in df.columns
    
    #  Boucle sur chaque ligne
    for idx, row in df.iterrows():
        if row['Zone'] != 'N':
            continue  # Ignorer les zones qui ne sont pas 'N'
            
        # Coordonnée X négative ?
        x_neg = row['X'] < 0  
        
        # Défense de l'équipe locale à gauche ?
        if home_team_D_side_exists:
            home_left = row['Home Team D Side'] == 'left'
        # Si la colonne 'Home Team D Side' n'existe pas (valeurs manquantes)
        else:
            if x_neg:
                home_left = False if row['Home Team Shot'] else True
            else:
                home_left = True if row['Home Team Shot'] else False
            
        # Logique ND/NO selon si tir de l'équipe locale ou visiteuse
        if row['Home Team Shot']:
            df.at[idx, 'Zone'] = 'ND' if (home_left and x_neg) or (not home_left and not x_neg) else 'NO'
        else:
            df.at[idx, 'Zone'] = 'NO' if (home_left and x_neg) or (not home_left and not x_neg) else 'ND'
    
    return df




"""
Cette fonction prend en argument un tir (événement) et calcule sa distance par rapport à la cage.
Ici on prend en compte la cage vers laquelle le tir à été effectué (direction du tir).
Les distances sont calculées en supposant que la trajectoire est une ligne droite vers la cage.
"""
def get_event_distance(event):

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
    
    # Calculer la distance entre la position du tir et les deux cages (Pythagore)
    dist_left = np.sqrt((x_coord - left_goal[0])**2 + (y_coord - left_goal[1])**2)
    dist_right = np.sqrt((x_coord - right_goal[0])**2 + (y_coord - right_goal[1])**2)

    # Choisir quelle distance est la correcte
    if zone == 'O' or zone == 'NO':
        shot_distance = min(dist_left, dist_right)
    elif zone == 'D' or zone == 'ND':
        shot_distance = max(dist_left, dist_right)
    else:   # zone = None (donnée manquante)
        shot_distance = min(dist_left, dist_right)
    
    return shot_distance.round(1)




"""
Cette fonction prend en argument une saison et ajoute un colonne indiauqnt 
la distance de chaque évènement par rapport aux filets.
On peut choisir si avoir les données sous forme précise ou arondie.
"""
def get_shots_distances(season, rounded=False):
    
    # Récupérer les données
    df = get_season_df(season) 

    # Calculer la distance
    df['Distance'] = df.apply(get_event_distance, axis=1)  
    
    # Éliminer les tirs avec type inconnu (type de tir manquant)
    df = df[df['Type of Shot'] != "Unknown"] 

    # Éliminer les tirs sans distance (coordonnées manquantes)
    df = df.dropna(subset=['Distance']) 
    
    if rounded:
        # Retourner aussi une version où les distances ont été arrondies à l'entier le plus proche
        rounded_df = df.copy()
        rounded_df['Distance'] = rounded_df['Distance'].round().astype(int)
        return rounded_df
    else:
        return df