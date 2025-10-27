from flask import Flask, request, jsonify
from flask_cors import CORS
from data_utils import read_dataset  
from ga_algorithm import GAAlgorithm  

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        file = request.files.get("file")
        target = request.form.get("target")

        if not file or not target:
            return jsonify({
                "error": "Please upload a dataset file and specify the target column."
            }), 400

        X, y = read_dataset(file, target)

        result = GAAlgorithm.GAOptimize(X, y)

        return jsonify({result })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
