from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow
import joblib
import json
from model.workout_recommender import FEATURES, MODEL_PATH, user_json, label_joblib, workout_json, user_act_json
from model.workout_recommender import predict_n, encode_hist_work, get_col_to_encode


app = Flask(__name__)

model = tensorflow.keras.models.load_model(MODEL_PATH, compile=False)
df_user = pd.read_json(user_json)

LABEL_ENCODER = joblib.load(label_joblib)


@app.route('/')
def index():
    return 'Hello World'


@app.route('/prediction/<user_id>', methods=['POST'])
def predict_workout(user_id):
    if request.method == 'POST':
        with open(workout_json, 'r') as f:
            workout_f = json.load(f)

        df_workout = pd.json_normalize(workout_f)
        df_user = pd.read_json(user_json)
        df_hist = pd.read_json(user_act_json)

        df_workout.drop(
            ['desc', 'jpg', 'gif', 'duration.desc', 'duration.min', 'duration.rep', 'duration.set', 'duration.sec'],
            axis=1, inplace=True
        )

        name = 'Thomas Lewis'

        user = df_user[df_user.name == name]
        gender_work = df_workout[
            (df_workout.gender == user.gender.values[0]) & (~df_workout.title.isin(df_hist[df_hist.name == name].title))
        ]

        n = 10
        top_n_prediction = predict_n(model, LABEL_ENCODER, n, name, gender_work, df_hist, user) # For now use `user` as dummy new user as the database is not updated in realtime
        df_prediction = gender_work.set_index('title').loc[top_n_prediction].reset_index()

        return jsonify(
            {
                'data': df_prediction.to_json()
            }
        )
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405


def load_user(user_id):
    pass


if __name__ == '__main__':
    app.run(debug=True)
