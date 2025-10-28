from flask import Flask, request, jsonify
from ga_algorithm import GAAlgorithm
from data_utils import read_dataset

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    target = request.form["target"]
    x,y=read_dataset(file,target)
    result = GAAlgorithm.GAOptimize(x, y)
    return jsonify(result)
if __name__ == "__main__":
    app.run(debug=True)


