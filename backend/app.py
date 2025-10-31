from flask import Flask, request, jsonify,send_file
from flask_cors import CORS
from data_utils import read_dataset  
from ga_algorithm import GAAlgorithm  
from traditional_algorithms.lasso_method import LassoAlgorithm
from traditional_algorithms.embedded_method import EmbeddedMethod
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

@app.route("/get_link_columns", methods=["POST"])
def get_link_columns():
    link = request.form.get("link")
    if not link:
        return jsonify({"error": "Please enter a dataset url."}), 400
    try:
        df = pd.read_csv(link, nrows=0)
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
       #ga algo
        result = GAAlgorithm.GAOptimize(x, y)
        selected_indices = result["selected_features_indices"]
        selected_features = [feature_names[i] for i in selected_indices if i < len(feature_names)]

        # lasso algo
        lasso_result = LassoAlgorithm.LassoOptimize(x, y)
        lasso_selected_indices = lasso_result["selected_features_indices"]
        lasso_selected_features = [feature_names[i] for i in lasso_selected_indices if i < len(feature_names)]
      
        #embeded method
        embd_result = EmbeddedMethod.run(x, y)
        embd_selected_indices = embd_result["selected_features_indices"]
        embd_selected_features = [feature_names[i] for i in embd_selected_indices if i < len(feature_names)]
       
        #statistical techniques
        st_result = StatsFeatureSelection.correlation_selection(x, y)
        st_selected_indices = st_result["selected_features_indices"]
        st_selected_features = [feature_names[i] for i in st_selected_indices if i < len(feature_names)]

        return jsonify({
            "bestFeatures": selected_features,
            "time": result["elapsed_time_seconds"],
            "accuracySelected": result["accuracy"]
        },
        {
            "bestFeatures": lasso_selected_features,
            "time": lasso_result["elapsed_time_seconds"],
            "accuracySelected": lasso_result["accuracy"]
        },
        {
            "bestFeatures": embd_selected_features,
            "time": embd_result["elapsed_time_seconds"],
            "accuracySelected": embd_result["accuracy"]
        },
        {
            "bestFeatures": st_selected_features,
            "time": st_result["elapsed_time_seconds"],
            "accuracySelected": st_result["accuracy"]
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route("/run_ga_direct", methods=["GET"])
def run_ga_direct():
    X, y = generate_dataset()
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
    df["Target"] = y
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    dataset_path = os.path.join(results_dir, "generated_dataset.csv")
    df.to_csv(dataset_path, index=False, encoding="utf-8")
    # ga algo
    result = GAAlgorithm.GAOptimize(X, y)
    column_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    selected_features = [column_names[i] for i in result["selected_features_indices"]]
    
    #lasso algo
    lasso_result = LassoAlgorithm.LassoOptimize(X, y)
    lasso_column_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    lasso_selected_features = [lasso_column_names[i] for i in lasso_result["selected_features_indices"]]
    
    # embedded method
    embd_result = EmbeddedMethod.run(X, y)
    embd_column_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    embd_selected_features = [embd_column_names[i] for i in embd_result["selected_features_indices"]]
    
    #statistical method
    st_result = StatsFeatureSelection.correlation_selection(X, y)
    st_column_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    st_selected_features = [st_column_names[i] for i in st_result["selected_features_indices"]]
    
    return jsonify({
        "bestFeatures": selected_features,
        "time": result["elapsed_time_seconds"],
        "accuracySelected": result["accuracy"]
    },
    {
        "bestFeatures": lasso_selected_features,
        "time": lasso_result["elapsed_time_seconds"],
        "accuracySelected": lasso_result["accuracy"]
    },
    {
        "bestFeatures": embd_selected_features,
        "time": embd_result["elapsed_time_seconds"],
        "accuracySelected": embd_result["accuracy"]
    },
    {
        "bestFeatures": st_selected_features,
        "time": st_result["elapsed_time_seconds"],
        "accuracySelected": st_result["accuracy"]
    })

# read from a link route
@app.route("/read_from_link", methods=["POST"])
def read_from_link() :
    try:
        link = request.form.get("link")
        target = request.form.get("target")

        if not link or not target:
            return jsonify({
                "error": "Please enter the url of dataset and specify the target column."
            }), 400
        df_header = pd.read_csv(link, nrows=0)
        if target not in df_header.columns:
            return jsonify({
                "error": f"The target column '{target}' does not exist in the dataset.",
                "availableColumns": list(df_header.columns)
            }), 400
        
        x, y = read_dataset(link, target)
        feature_names = [col for col in df_header.columns if col != target]
       #ga algo
        result = GAAlgorithm.GAOptimize(x, y)
        selected_indices = result["selected_features_indices"]
        selected_features = [feature_names[i] for i in selected_indices if i < len(feature_names)]

        # lasso algo
        lasso_result = LassoAlgorithm.LassoOptimize(x, y)
        lasso_selected_indices = lasso_result["selected_features_indices"]
        lasso_selected_features = [feature_names[i] for i in lasso_selected_indices if i < len(feature_names)]
      
        #embeded method
        embd_result = EmbeddedMethod.run(x, y)
        embd_selected_indices = embd_result["selected_features_indices"]
        embd_selected_features = [feature_names[i] for i in embd_selected_indices if i < len(feature_names)]
       
        #statistical techniques
        st_result = StatsFeatureSelection.correlation_selection(x, y)
        st_selected_indices = st_result["selected_features_indices"]
        st_selected_features = [feature_names[i] for i in st_selected_indices if i < len(feature_names)]

        return jsonify({
            "bestFeatures": selected_features,
            "time": result["elapsed_time_seconds"],
            "accuracySelected": result["accuracy"]
        },
        {
            "bestFeatures": lasso_selected_features,
            "time": lasso_result["elapsed_time_seconds"],
            "accuracySelected": lasso_result["accuracy"]
        },
        {
            "bestFeatures": embd_selected_features,
            "time": embd_result["elapsed_time_seconds"],
            "accuracySelected": embd_result["accuracy"]
        },
        {
            "bestFeatures": st_selected_features,
            "time": st_result["elapsed_time_seconds"],
            "accuracySelected": st_result["accuracy"]
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
    
@app.route("/download_generated_dataset", methods=["GET"])
def download_generated_dataset():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    dataset_path = os.path.join(results_dir, "generated_dataset.csv")
    if not os.path.exists(dataset_path):
        return jsonify({"error": "Dataset not found. Generate it first."}), 404
    return send_file(dataset_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)