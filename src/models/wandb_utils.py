import os
import wandb
from dotenv import load_dotenv
import joblib

class WandbLogger:
    def __init__(self, project_name, run_name, config=None):
        load_dotenv()
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        self.run = wandb.init(project=project_name, name=run_name, config=config, reinit='finish_previous')

    def log_metrics(self, metrics_dict, step=None, as_summary=False):
        """Log des métriques dans wandb."""
        if step is not None:
            self.run.log(metrics_dict, step=step)
        else:
            wandb.log(metrics_dict)
        if as_summary:
            for k, v in metrics_dict.items():
                self.run.summary[k] = v

    def log_hyperparameters(self, hyperparams_dict):
        """Met à jour les hyperparamètres dans wandb."""
        self.run.config.update(hyperparams_dict)

    def log_selected_features(self, features_list):
        """
        Log the list of selected features
        """
        self.run.summary["n_selected_features"] = len(features_list)

        table = wandb.Table(columns=["feature"])
        for f in features_list:
            table.add_data(f)

        self.run.log({"selected_features": table})

    def log_model_artifact(self, model, artifact_name, artifact_type, description=""):
        model_path = f"{artifact_name}.pkl"
        joblib.dump(model, model_path)

        # Création et envoi de l'artifact
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type, description=description)
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)

        # Suppression du fichier local temporaire
        os.remove(model_path)

    def log_figure(self, name: str, fig):
        """Logs figure to wandb."""
        self.run.log({name: wandb.Image(fig)})

    def finish(self):
        """Termine proprement la session W&B."""
        if self.run is not None and not self.run.finish:
            self.run.finish()


    

