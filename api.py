import utils.app_utils as app_utils
import utils.data_process as dp
from flask import Flask, jsonify, request, abort


api = Flask(__name__)

# Load the model globally. No need to reload it every time.
data_path = "data/CONLL2003/"
model_path = "model/"

word2idx_loaded = dp.file_to_dict(model_path + "train_index.json")
idx2tag_loaded = dp.file_to_dict(model_path + "tag_index.json", int_key=True)

loaded_model = app_utils.load_model(model_path + "bilstm.pt", word2idx_loaded, idx2tag_loaded)


@api.route("/api/predict", methods=["POST"])
def return_prediction():
    if not request.json or not 'sentence' in request.json:
        abort(400)
    sentence = request.json["sentence"]
    pred = app_utils.predict_on_sentence(sentence, loaded_model, word2idx_loaded, idx2tag_loaded)
    return jsonify({"prediction:": pred})


if __name__ == '__main__':
    api.run(debug=True, host='0.0.0.0', port='8000')