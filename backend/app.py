from flask import Flask, request, jsonify,send_file
from flask_cors import CORS
from data_utils import read_dataset  
from ga_algorithm import GAAlgorithm  
import os
app = Flask(__name__)
CORS(app)
@app.route("/")
def index():
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'index.html')
    return send_file(frontend_path)

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


