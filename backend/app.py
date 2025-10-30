from flask import Flask, request, jsonify,send_file
from flask_cors import CORS
from data_utils import read_dataset  
from ga_algorithm import GAAlgorithm  
from traditional_algorithms.embedded_method import EmbeddedMethod
from traditional_algorithms.lasso_method import LassoAlgorithm
from statistical_techniques.statistical_techniques import StatsFeatureSelection
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
        embded=EmbeddedMethod.run(x,y)
        lasso=LassoAlgorithm.LassoOptimize(x,y)
        statsf=StatsFeatureSelection.correlation_selection(x,y)
        selected_indices = result["selected_features_indices"]
        selected_features = [feature_names[i] for i in selected_indices if i < len(feature_names)]

        #just for testing results

        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        result_path = os.path.join(results_dir, "all_results.txt")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write("========== FEATURES SELECTION RESULTS ==========\n\n")

            # GA Algorithm
            ga_indices = result.get("selected_features_indices", [])
            ga_features = [feature_names[i] for i in ga_indices if i < len(feature_names)]
            f.write("=== Genetic Algorithm (GA) ===\n")
            f.write(f"Selected feature indices: {ga_indices}\n")
            f.write(f"Selected feature names: {ga_features}\n")
            f.write(f"Number of selected features: {result.get('num_selected_features', 'N/A')}\n")
            f.write(f"Accuracy: {result.get('accuracy', 0):.4f}\n")
            f.write(f"Elapsed time (s): {result.get('elapsed_time_seconds', 0):.4f}\n\n")

            #  Embedded Method
            emb_indices = embded.get("selected_features_indices", [])
            emb_features = [feature_names[i] for i in emb_indices if i < len(feature_names)]
            f.write("=== Embedded Method ===\n")
            f.write(f"Selected feature indices: {emb_indices}\n")
            f.write(f"Selected feature names: {emb_features}\n")
            f.write(f"Number of selected features: {embded.get('num_selected_features', 'N/A')}\n")
            f.write(f"Accuracy: {embded.get('accuracy', 0):.4f}\n")
            f.write(f"Elapsed time (s): {embded.get('elapsed_time_seconds', 0):.4f}\n\n")

            #  Lasso Algorithm
            lasso_indices = lasso.get("selected_features_indices", [])
            lasso_features = [feature_names[i] for i in lasso_indices if i < len(feature_names)]
            f.write("=== Lasso Algorithm ===\n")
            f.write(f"Selected feature indices: {lasso_indices}\n")
            f.write(f"Selected feature names: {lasso_features}\n")
            f.write(f"Number of selected features: {lasso.get('num_selected_features', 'N/A')}\n")
            f.write(f"Accuracy: {lasso.get('accuracy', 0):.4f}\n")
            f.write(f"Elapsed time (s): {lasso.get('elapsed_time_seconds', 0):.4f}\n\n")

            #  Statistical Feature Selection (Correlation)
            stats_indices = statsf.get("selected_features_indices", [])
            stats_features = [feature_names[i] for i in stats_indices if i < len(feature_names)]
            f.write("=== Statistical Feature Selection (Correlation) ===\n")
            f.write(f"Selected feature indices: {stats_indices}\n")
            f.write(f"Selected feature names: {stats_features}\n")
            f.write(f"Number of selected features: {statsf.get('num_selected_features', 'N/A')}\n")
            f.write(f"Accuracy: {statsf.get('accuracy', 0):.4f}\n")
            f.write(f"Elapsed time (s): {statsf.get('elapsed_time_seconds', 0):.4f}\n\n")

            f.write("========== END OF RESULTS ==========\n")

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