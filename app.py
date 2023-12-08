import json

import joblib
import numpy as np
import pandas as pd
import tensorflow
from flask import Flask, jsonify, request
from sklearn.preprocessing import LabelEncoder

from database import get_user_df, get_hist_work_df, open_connection
from model.workout_recommender import (FEATURES, MODEL_PATH, encode_hist_work,
                                        get_col_to_encode, label_joblib,
                                        predict_n, user_act_json, user_json,
                                        workout_json)

app = Flask(__name__)

model = tensorflow.keras.models.load_model(MODEL_PATH, compile=False)
df_user = pd.read_json(user_json)

LABEL_ENCODER = joblib.load(label_joblib)

with open(workout_json, 'r') as f:
    workout_f = json.load(f)


@app.route('/')
def index():
    return 'Hello World'


@app.route('/workout_prediction/<user_id>', methods=['POST', 'GET'])
def predict_workout(user_id):
    user_id = '06cbf213-0090-43aa-ba95-7aa693da68d1' # DUSMDAFNDFKJSHBABdhfadsvgDFAKDSBFHDSBF

    if request.method == 'GET':
        connection = open_connection()
        
        df_workout = pd.json_normalize(workout_f)
        df_user = get_user_df(connection, user_id)
        df_hist = get_hist_work_df(connection, user_id)
        
        connection.close()

        gender_work = df_workout[
            (df_workout.gender == df_user.gender.values[0]) & (~df_workout.ExerciseId.isin(df_hist[df_hist.UserId == df_user.id.values[0]].ExerciseId))
        ]

        n = 10
        top_n_prediction = predict_n(model, LABEL_ENCODER, n, gender_work, df_hist, df_user) # For now use `user` as dummy new user as the database is not updated in realtime
        df_prediction = gender_work.set_index('title').loc[top_n_prediction].reset_index()

        return jsonify(df_prediction.to_json())
    else:
        return jsonify({
            'status': {
                'code': 405,
                'message': 'Method not allowed'
            },
            'data': None,
        }), 405


if __name__ == '__main__':
    app.run(debug=True)
