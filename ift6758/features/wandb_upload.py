import pandas as pd
import wandb

#Load le training dataset
df = pd.read_csv('../../games_data/feature_dataset_2_train.csv')

#filre pour la partie specifique
game_id = '2017021065'
df['Game ID'] = df['Game ID'].astype(str) #ensure same type (think in csv its int)
df_filtered = df[df['Game ID'] == game_id].copy()

print(f'Found {len(df_filtered)} events for game {game_id}')

#initialize wandb
run = wandb.init(project="IFT6758.2025-A03", job_type='dataset-upload', name='dataset_2017021065')

#cree artefact
artifact = wandb.Artifact(
    'wpg_v_wsh_2017021065',
    type='dataset'
)

#Ajoute dataset en tant que table
my_table = wandb.Table(dataframe=df_filtered)
artifact.add(my_table, 'wpg_v_wsh_2017021065')

#Log artifact
run.log_artifact(artifact)
run.finish()

print('Upload Complete')

