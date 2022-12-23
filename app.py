import numpy as np
import lightgbm
from flask import Flask, request, jsonify, render_template, abort, url_for

import base64
from io import BytesIO
from matplotlib.figure import Figure

app = Flask(__name__)
model = lightgbm.Booster(model_file="model/model.txt")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_features = request.form['text']
    input_features = input_features.strip().split(',')
    input_features = np.array(input_features).astype(float)

    # raise exception when dimension of data is incorrect
    if len(input_features) != model.num_feature():
        return render_template('index.html',
                               prediction_text='Make sure you have entered 28 numerical values separated by commas',
                               )

    # make prediction using the model
    # reshape the input to fit into the model
    input_features = input_features.reshape((1, -1))
    prediction = round(model.predict(input_features)[0])

    # make a graph using the input data
    fig = Figure(facecolor='black', figsize=(7, 5))
    ax = fig.subplots()
    ax.set_facecolor("black")
    ax.get_xaxis().set_visible(False)

    ax.set_ylabel('Sales')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='y', colors='white')

    barlist = ax.bar(
        range(len(input_features[0])+1), np.append(input_features[0], [prediction]))
    barlist[-1].set_color('tab:orange')

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")

    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return render_template('index.html',
                           prediction_text='The upcoming month sales should be {}'.format(
                               prediction),
                           sales_graph=f"<img src='data:image/png;base64,{data}'/>"
                           )


@app.route('/api/predict', methods=['POST'])
def batch_predict():
    request_body = request.get_json()
    data = np.array(request_body['data'])

    # raise exception when dimension of data is larger than 2-D
    if data.ndim > 2 or data.shape[-1] != model.num_feature():
        abort(
            400, f"Input data should be in 1D/2D shape and contain exactly {model.num_feature()} features")

    # reformat data into 2D in case it is 1-D
    if data.ndim == 1:
        data = data.reshape((1, -1))

    predictions = model.predict(data)
    response = {}
    response['result'] = list(predictions)

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
