import json

import joblib
import numpy as np
import pandas as pd
import tensorflow
from flask import Flask, jsonify, request
from sklearn.preprocessing import LabelEncoder

from database import get_user_df, get_hist_work_df, open_connection, get_user_bmi_df
from model.workout_recommender import (MODEL_PATH as WORK_MODEL_PATH, label_joblib as work_label_joblib,
                                        work_predict_n, user_json, workout_json)
from model.nutrition_recommender import MODEL_PATH as NUTRITION_MODEL_PATH, label_joblib as nutrition_label_joblib, nutrition_predict


app = Flask(__name__)

WORK_MODEL = tensorflow.keras.models.load_model(WORK_MODEL_PATH, compile=False)
NUTRITION_MODEL = tensorflow.keras.models.load_model(NUTRITION_MODEL_PATH, compile=False)

WORK_LABEL_ENCODER = joblib.load(work_label_joblib)
NUTRITION_LABEL_ENCODER = joblib.load(nutrition_label_joblib)

with open(workout_json, 'r') as f:
    workout_f = json.load(f)


@app.route('/')
def index():
    return 'Hello World'


@app.route('/workout_prediction/<user_id>', methods=['POST']) # Should change to json request
def predict_workout(user_id):
    user_id = '9be2e512-8645-4c8b-b54b-a6823d65dd5a' # DUSMDAFNDFKJSHBABdhfadsvgDFAKDSBFHDSBF

    if request.method == 'POST':
        connection = open_connection()
        
        df_workout = pd.json_normalize(workout_f)
        df_user = get_user_df(connection, user_id)
        df_hist = get_hist_work_df(connection, user_id)
        
        connection.close()

        gender_work = df_workout[
            (df_workout.gender == df_user.gender.values[0]) & (~df_workout.title.isin(df_hist[df_hist.UserId == df_user.id.values[0]].ExerciseId)) # Should change
        ]

        n = 10
        top_n_prediction = work_predict_n(WORK_MODEL, WORK_LABEL_ENCODER, n, gender_work, df_user)
        df_prediction = gender_work.set_index('title').loc[top_n_prediction].reset_index()

        return df_prediction.to_json()
    else:
        return jsonify({
            'status': {
                'code': 405,
                'message': 'Method not allowed'
            },
            'data': None,
        }), 405


@app.route('/nutrition_prediction/<user_id>', methods=['POST']) # Should change to json request, use correct field name on df
def predict_nutrition(user_id):
    user_id = '9be2e512-8645-4c8b-b54b-a6823d65dd5a' # DUSMDAFNDFKJSHBABdhfadsvgDFAKDSBFHDSBF

    if request.method == 'POST':
        connection = open_connection()

        df_user = get_user_bmi_df(connection, user_id)
        
        connection.close()

        prediction = nutrition_predict(NUTRITION_MODEL, df_user, NUTRITION_LABEL_ENCODER)

        return jsonify(prediction)
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
