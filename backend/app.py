from flask import Flask, request, jsonify,send_file
from flask_cors import CORS
from data_utils import read_dataset  
from ga_algorithm import GAAlgorithm  
from generate_random_dataset import generate_dataset
import os
import pandas as pd

app = Flask(__name__)
CORS(app)
@app.route("/")
def index():
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'index.html')
    return send_file(frontend_path)

@app.route("/<filename>")
def serve_static_files(filename):
    if '.' in filename:
        frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
        filepath = os.path.join(frontend_dir, filename)
        
        try:
            if os.path.exists(filepath):
                return send_file(filepath)
            else:
                return f"<h1>File not found: {filename}</h1>", 404
        except Exception as e:
            return f"<h1>Error serving {filename}: {str(e)}</h1>", 500
            
    return "File not found", 404

@app.route("/get_columns", methods=["POST"])
def get_columns():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Please upload a dataset file."}), 400
    try:
        df = pd.read_csv(file, nrows=0)
        columns = list(df.columns)
        
        return jsonify({"columns": columns})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        file = request.files.get("file")
        target = request.form.get("target")

        if not file or not target:
            return jsonify({
                "error": "Please upload a dataset file and specify the target column."
            }), 400
        df_header = pd.read_csv(file, nrows=0)
        if target not in df_header.columns:
            return jsonify({
                "error": f"The target column '{target}' does not exist in the uploaded dataset.",
                "availableColumns": list(df_header.columns)
            }), 400
        
        file.seek(0)
        x, y = read_dataset(file, target)
        feature_names = [col for col in df_header.columns if col != target]

        result = GAAlgorithm.GAOptimize(x, y)

        selected_indices = result["selected_features_indices"]
        selected_features = [feature_names[i] for i in selected_indices if i < len(feature_names)]

        return jsonify({
            "bestFeatures": selected_features,
            "time": result["elapsed_time_seconds"],
            "accuracySelected": result["accuracy"]
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route("/run_ga_direct", methods=["GET"])
def run_ga_direct():
    X, y = generate_dataset()
    result = GAAlgorithm.GAOptimize(X, y)
    column_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    selected_features = [column_names[i] for i in result["selected_features_indices"]]
    
    return jsonify({
        "bestFeatures": selected_features,
        "time": result["elapsed_time_seconds"],
        "accuracySelected": result["accuracy"]
    })

if __name__ == "__main__":
    app.run(debug=True)

