from flask import Flask, request, jsonify,send_file
from flask_cors import CORS
from data_utils import read_dataset  
from ga_algorithm import GAAlgorithm  
import json
import os
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

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        file = request.files.get("file")
        target = request.form.get("target")

        if not file or not target:
            return jsonify({
                "error": "Please upload a dataset file and specify the target column."
            }), 400

        x, y = read_dataset(file, target)
        
        result = GAAlgorithm.GAOptimize(x, y)
        output_path = os.path.join(os.path.dirname(__file__), "ga_result.txt")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)

        

        return jsonify({
    "bestFeatures": result["selected_features_indices"],
    "time": result["elapsed_time_seconds"],
    "accuracySelected": result["accuracy"]
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)


