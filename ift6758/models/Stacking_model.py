import os
import sys
import pandas as pd
import joblib
import wandb
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from dotenv import load_dotenv
load_dotenv()

from CatBoost_model import CatBoostModel
from LightGBM_model import LightGBMModel
sys.path.append(os.path.abspath("../.."))
from ift6758.models.wandb_utils import WandbLogger


class StackingModel:
    def __init__(self, estimators, meta_model):
        # Logger for Wandb run
        self.logger = WandbLogger(project_name="IFT6758.2025-A03", run_name='Stacking(MLP+CatBoost+LightGBM)')

        self.estimators = estimators  # Base models
        self.meta_model = meta_model

    """Train the meta-model only on estimators (base models) predictions"""
    def train(self, X_train, y_train):
        # Get predictions from all base models
        base_predictions = []
        for name, model in self.estimators:
            preds = model.predict_proba(X_train)
            base_predictions.append(preds)

        X_meta = np.column_stack(base_predictions)

        # Train meta-model with base models predictions as features
        self.meta_model.fit(X_meta, y_train)

    def predict(self, X):
        # Get predictions from base models
        base_predictions = []
        for name, model in self.estimators:
            preds = model.predict_proba(X)
            base_predictions.append(preds)

        X_meta = np.column_stack(base_predictions)
        # Meta-model prediction
        return self.meta_model.predict(X_meta)

    def predict_proba(self, X):
        # Get predictions from base models
        base_predictions = []
        for name, model in self.estimators:
            preds = model.predict_proba(X)
            base_predictions.append(preds)

        X_meta = np.column_stack(base_predictions)
        # Meta-model prediction probability
        return self.meta_model.predict_proba(X_meta)[:, 1]

    def evaluate(self, X_valid, y_valid):
        y_pred = self.predict(X_valid)
        y_prob = self.predict_proba(X_valid)

        metrics = {
            "accuracy": accuracy_score(y_valid, y_pred),
            "precision": precision_score(y_valid, y_pred),
            "recall": recall_score(y_valid, y_pred),
            "f1_score": f1_score(y_valid, y_pred),
            "roc_auc": roc_auc_score(y_valid, y_prob)
        }
        return metrics


    def get_ROC(self, X_valid, y_valid):
        y_prob = self.predict_proba(X_valid)
        fpr, tpr, _ = roc_curve(y_valid, y_prob)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc, y_prob

    @staticmethod
    def get_goal_rate_by_percentile(y_true, y_prob):
        df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
        df['centile'] = pd.qcut(df['y_prob'], q=100, labels=False, duplicates='drop')
        goal_rate = df.groupby('centile')['y_true'].mean()
        return goal_rate.index, goal_rate.values

    @staticmethod
    def get_cumulative_goal(y_true, y_prob):
        df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
        df = df.sort_values('y_prob', ascending=False)
        df['But cumulatifs'] = df['y_true'].cumsum()
        df['Ratio cumulatifs'] = df['But cumulatifs'] / df['y_true'].sum()
        df['Proportion de tirs'] = np.arange(1, len(df) + 1) / len(df)
        return df['Proportion de tirs'], df['Ratio cumulatifs']

    @staticmethod
    def get_calibration(y_true, y_prob, n_bins=10):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        return prob_pred, prob_true


"""Load trained base models from Wandb"""
def load_models(models):
    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
    loaded_models = []
    for name, artifact_url in models.items():
        artifact = api.artifact(artifact_url)
        # Download artifact to a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        artifact_path = artifact.download(root=temp_dir.name)
        model_file_name = [f for f in os.listdir(artifact_path) if f.endswith(".pkl")]
        model_file = os.path.join(artifact_path, model_file_name[0])
        # Load model
        if model_file.endswith(".pkl"):
            # Save loaded model
            loaded_models.append((name, joblib.load(model_file)))
            temp_dir.cleanup()
        else:
            raise ValueError("Model file not .pkl")
    return loaded_models


if __name__ == "__main__":
    name = 'Stacking(MLP+CatBoost+LightGBM)'
    # meta_model_name = 'LogisticRegression'
    # meta_model_name = 'Calibrated RidgeClassifier'
    meta_model_name = 'XGBoostClassifier'

    print(name)

    # Models
    models = {
        'MLP': 'IFT6758--2025-A03/IFT6758.2025-A03/MLPClassifier:v9',
        'CatBoost': 'IFT6758--2025-A03/IFT6758.2025-A03/CatBoostClassifier:v12',
        'LightGBM': 'IFT6758--2025-A03/IFT6758.2025-A03/LightGBM:v2',
    }

    # Load models
    estimators = load_models(models)

    # Load Dataset
    data = pd.read_csv('../../games_data/feature_dataset_2_train.csv')
    y = data['is_goal']
    X = data.drop(columns=['is_goal'])

    # Data split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Meta model
    meta_model = None

    if meta_model_name == 'LogisticRegression':
        meta_model = LogisticRegression(max_iter=500, random_state=42)

    if meta_model_name == 'Calibrated RidgeClassifier':
        base_ridge = RidgeClassifier(alpha=1.0, random_state=42)
        meta_model = CalibratedClassifierCV(base_ridge, method='sigmoid', cv=3)

    if meta_model_name == 'XGBoostClassifier':
        meta_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )

    # Staking model
    stacked_model = StackingModel(estimators, meta_model)

    # Train
    stacked_model.train(X_train, y_train)

    # Evaluate
    metrics = stacked_model.evaluate(X_valid, y_valid)
    print("Metrics:", metrics)

    """
    Plot the curves (individual figures)
    """

    # Get predictions
    fpr, tpr, roc_auc, y_prob = stacked_model.get_ROC(X_valid, y_valid)

    result = {
        name: {
            'roc': (fpr, tpr, roc_auc),
            'goal_rate': stacked_model.get_goal_rate_by_percentile(y_valid, y_prob),
            'cumulative': stacked_model.get_cumulative_goal(y_valid, y_prob),
            'calibration': stacked_model.get_calibration(y_valid, y_prob)
        }
    }

    # --- ROC ---
    roc_fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title(f'Courbe ROC - {name}')
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.legend()
    ax.grid(True)

    # --- Goal Rate by Percentile ---
    x_g, y_g = result[name]['goal_rate']
    y_smooth = pd.Series(y_g).rolling(window=5, center=True, min_periods=1).mean()

    goal_rate_fig, ax = plt.subplots()
    ax.plot(x_g, y_smooth, label=name)
    ax.invert_xaxis()
    ax.set_title(f'Taux de buts par centile - {name}')
    ax.set_xlabel('Centile de probabilité')
    ax.set_ylabel('Taux de buts')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    # --- Cumulative Goals ---
    x_c, y_c = result[name]['cumulative']

    cumulative_goals_fig, ax = plt.subplots()
    ax.plot(x_c, y_c, label=name)
    ax.set_title(f'Proportion cumulée des buts - {name}')
    ax.set_xlabel('Proportion de tirs')
    ax.set_ylabel('Proportion de buts cumulée')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(1 - x) * 100:.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    ax.legend()
    ax.grid(True)

    # --- Calibration / Reliability ---
    x, y = result[name]['calibration']

    calibration_fig, ax = plt.subplots()
    ax.plot(x, y, color='blue', label=name)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    ax.set_title(f'Courbe de calibration/fiabilité - {name}')
    ax.set_xlabel('Probabilité prédite')
    ax.set_ylabel('Fréquence observée')
    ax.legend()
    ax.grid(True)

    """
    Plot the curves (unique figure)
    """

    summary_fig, axs = plt.subplots(2, 2, figsize=(18, 10))

    # --- ROC Curve ---
    fpr, tpr, roc_auc = result[name]['roc']
    axs[0, 0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
    axs[0, 0].plot([0, 1], [0, 1], 'k--')
    axs[0, 0].set_title('Courbe ROC')
    axs[0, 0].set_xlabel("Taux de faux positifs")
    axs[0, 0].set_ylabel("Taux de vrais positifs")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # --- Goal Rate by Percentile ---
    x_g, y_g = result[name]['goal_rate']
    y_smooth = pd.Series(y_g).rolling(window=5, center=True, min_periods=1).mean()
    axs[0, 1].plot(x_g, y_smooth, label=name)
    axs[0, 1].invert_xaxis()
    axs[0, 1].set_title('Taux de buts par centile')
    axs[0, 1].set_xlabel('Centile')
    axs[0, 1].set_ylabel('Taux de buts')
    axs[0, 1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # --- Cumulative Goals ---
    x_c, y_c = result[name]['cumulative']
    axs[1, 0].plot(x_c, y_c, label=name)
    axs[1, 0].set_title('Proportion cumulée de buts')
    axs[1, 0].set_xlabel('Proportion de tirs')
    axs[1, 0].set_ylabel('Proportion cumulée de buts')
    axs[1, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(1 - x) * 100:.0f}"))
    axs[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # --- Calibration / Reliability ---
    x, y = result[name]['calibration']
    axs[1, 1].plot(x, y, color='blue', label=name)
    axs[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.7)
    axs[1, 1].set_title(f'Courbe de calibration/fiabilité - {name}')
    axs[1, 1].set_xlabel('Probabilité prédite')
    axs[1, 1].set_ylabel('Fréquence observée')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


    # --- Log metrics and figures ---
    # Metrics
    stacked_model.logger.log_metrics(metrics, as_summary=True)
    stacked_model.logger.log_hyperparameters(stacked_model.meta_model.get_params())
    description = f'{name}: meta-model={meta_model_name}'
    # Artifact
    clean_estimators = []
    for name, model_wrapper in estimators:
        # Save base models wrapper
        clean_estimators.append((name, model_wrapper))
    # Remove the logger from each wrapper
    for name, wrapper in clean_estimators:
        if hasattr(wrapper, 'logger'):
            delattr(wrapper, 'logger')
    model_to_save = {
        'estimators': clean_estimators,
        'meta_model': stacked_model.meta_model
    }
    stacked_model.logger.log_model_artifact(model_to_save, f'StackingModel', 'model', description)

    # Figures
    stacked_model.logger.log_figure(f'ROC_{name}', roc_fig)
    stacked_model.logger.log_figure(f'GoalRate_{name}', goal_rate_fig)
    stacked_model.logger.log_figure(f'CumulativeGoals_{name}', cumulative_goals_fig)
    stacked_model.logger.log_figure(f'Calibration_{name}', calibration_fig)
    stacked_model.logger.log_figure(f'Summary_{name}', summary_fig)

    plt.close(roc_fig)
    plt.close(goal_rate_fig)
    plt.close(cumulative_goals_fig)
    plt.close(calibration_fig)
    plt.close(summary_fig)

    stacked_model.logger.finish()