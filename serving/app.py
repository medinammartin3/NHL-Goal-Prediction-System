"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:

    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
# Project name: "IFT6758.2025-A03"
# Logistic regression distance artifact: LogReg_Model_with_distance
# Logistic regression distance+Angle artifact: LogReg_Model_with_distance_and_angle
#entity: IFT6758--2025-A03 (team name pas notre username)

#To run this, open terminal to launch server than do the operations in another terminal, while still running the server.
import os
import sys
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
import wandb
from dotenv import load_dotenv

# Ajoute le repertoire racine du projet au pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

#Log file location
LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
MODELS_DIR = "models" #Directory where downloaded models will be stored

app = Flask(__name__) #Initialize flask app

#Variable globale pour le gestionnaire de modeles
current_model = None
current_model_name = None
current_features = None
# Comme mon ordi est un windows, jutilise waitress donc je mets le code d'initialisation ici
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

#Configuring logging both to file and to console
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

app.logger.info("Flask service starting.")
os.makedirs(MODELS_DIR, exist_ok=True)


#Cant use this apparently since windows have to use waitress not gunicorn, but if use gunicorn (should work)
"""@app.before_first_request
def before_first_request():
    # Put 3*"
    #Hook to handle any initialization before the first request (e.g. load model,
    #setup logging handler, etc.)
    # Put 3*"


    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    app.logger.info("Flask service starting.")
    os.makedirs(MODELS_DIR, exist_ok=True)"""


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    app.logger.info("GET request received on /logs endpoint.")

    #If log file does not exist yet, return an empty response
    if not os.path.exists(LOG_FILE):
        app.logger.warning(f"Log file {LOG_FILE} does not exist yet")
        return jsonify({"logs": [], "message": "No logs yet"}), 200

    try:
        #Read full log content
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()

        app.logger.info(f"Successfully read {len(lines)} log lines")

        return jsonify({
            "logs": lines[-100:], #return last 100 lines
            "total": len(lines)
        })
    # In case of a problem, we log the error
    except Exception as e:
        app.logger.error(f"Error reading log file: {e}")
        return jsonify({"error": str(e)}), 500



@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model
    """

    global current_model, current_model_name, current_features

    # Parse incoming POST request data
    data = request.get_json()
    app.logger.info(f"POST request received on /download_registry_model with data: {data}")

    # Required parameters to locate a model in Wandb
    entity = data.get("entity")
    project = data.get("project")
    artifact_name = data.get("artifact_name")
    version = data.get("version", "latest")

    # Basic validation
    if not (entity and project and artifact_name):
        app.logger.error("Missing required parameters: entity, project or artifact_name")
        return jsonify({"success": False, "error": "Missing parameters"}), 400

    # Fully qualified WandB artifact reference
    full_name = f"{entity}/{project}/{artifact_name}:{version}"
    local_path = os.path.join(MODELS_DIR, f"{artifact_name.replace('/', '_')}_{version}.joblib")
    app.logger.info(f"Attempting to load model: {full_name}")
    app.logger.info(f"Local path: {local_path}")

    # Check if model already exists locally
    if os.path.exists(local_path):
        try:
            app.logger.info(f"Model already exists locally: {local_path}")
            #Keep track of the old model for logging purpose.
            previous_model = current_model_name
            #Use joblib to load the model in memory
            current_model = joblib.load(local_path)
            current_model_name = full_name
            app.logger.info(f"MODEL CHANGE: Successfully loaded existing model from local storage.")
            app.logger.info(f"Previous model: {previous_model}")
            app.logger.info(f"New model: {current_model_name}")

            try:
                if hasattr(current_model.model, "feature_names_in_"):
                    current_features = list(current_model.model.feature_names_in_)
                    app.logger.info(f"[AUTO] Model features detected: {current_features}")
                else:
                    app.logger.warning("Model has no feature_names_in_ attribute — keeping previous features.")
            except Exception as e:
                app.logger.error(f"Could not detect model features: {e}")

            return jsonify({"success": True, "loaded_local": True, "model": full_name, 'features': current_features})
        except Exception as e:
            #If the file exists but is corrupted or unreadable, log the failure
            app.logger.error(f"Failed to load local model from {local_path}: {e}")
            return jsonify({"success": False, "error": f"Local model exists but failed to load: {str(e)}"}), 500

    # Download from WandB if not found locally
    try:
        app.logger.info(f"Model not found locally, downloading model from WandB: {full_name}")
        api = wandb.Api()
        artifact = api.artifact(full_name)
        app.logger.info(f"Artifact found in WandB registry, starting download...")
        artifact_dir = artifact.download()
        app.logger.info(f"Download completed to directory: {artifact_dir}")

        # Find the .joblib file inside artifact
        for file in os.listdir(artifact_dir):
            if file.endswith(".joblib"):
                src = os.path.join(artifact_dir, file)
                # Copy model to our unified location
                joblib.dump(joblib.load(src), local_path)
                # Load model into memory
                current_model = joblib.load(local_path)
                current_model_name = full_name

        try:
            if hasattr(current_model.model, "feature_names_in_"):
                current_features = list(current_model.model.feature_names_in_)
                app.logger.info(f"[AUTO] Model features detected: {current_features}")
            else:
                app.logger.warning("Model has no feature_names_in_ attribute — keeping previous features.")
        except Exception as e:
            app.logger.error(f"Could not detect model features: {e}")

        app.logger.info(f"Model loaded successfully: {full_name}")
        return jsonify({"success": True, "model": full_name, "features": current_features})

    except Exception as e:
        app.logger.error(f"Failed to download model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    global current_model #we only need to read the loaded model

    app.logger.info("POST request received on /predict endpoint")

    #Ensure a model is loaded before handling predictions
    if current_model is None:
        app.logger.info("No model loaded. Please load a model using /download_registry_model first.")
        return jsonify({"error": "No model loaded"}), 503

    data = request.get_json()

    if data is None:
        app.logger.info("No data provided in request body")
        return jsonify({"error": "No data provided"}), 400

    try:
        # Input must be a dict of lists matching training features
        X = pd.DataFrame.from_dict(data)
        app.logger.info("Received data for prediction.")

    except Exception as e:
        app.logger.info(f"Invalid JSON format: {str(e)}")
        return jsonify({"error": "Invalid JSON format", "details": str(e)}), 400

    try:
        # Prefer predict_proba if available (classification model)
        if hasattr(current_model, "predict_proba"):
            app.logger.info("Using predict_proba() for predictions.")
            #Convert predictions to a list so it can be JSON
            preds = current_model.predict_proba(X).tolist()
        else:
            #Fallback for models without predict_proba
            app.logger.info("Using predict() for predictions (predict_proba not available)")
            preds = current_model.predict(X).tolist()

        app.logger.info(f"PREDICTION SUCCESS: Generated {len(preds)} predictions using model {current_model_name}")

        #Construct the final JSON response with the results
        response = {
            "predictions": preds,
            "n_samples": len(preds),
            "model": current_model_name
        }

        return jsonify(response)

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        app.logger.error(error_msg)
        app.logger.error(f"Error type: {type(e).__name__}")

        return jsonify({
            "error": "Prediction failed",
            "details": str(e),
            "model": current_model_name
        }), 500