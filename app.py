import os
from datetime import date, timedelta

import joblib
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, request

from database import (get_hist_work_df, get_user_bmi_df, get_user_df,
                      open_connection)
from model.nutrition_recommender import MODEL_PATH as NUTRITION_MODEL_PATH
from model.nutrition_recommender import label_joblib as nutrition_label_joblib
from model.nutrition_recommender import nutrition_predict
from model.workout_recommender import MODEL_PATH as WORK_MODEL_PATH
from model.workout_recommender import WORKOUT_DROP
from model.workout_recommender import label_joblib as work_label_joblib
from model.workout_recommender import work_predict_n, workout_json


app = Flask(__name__)

WORK_MODEL = tf.keras.models.load_model(WORK_MODEL_PATH, compile=False)
NUTRITION_MODEL = tf.keras.models.load_model(NUTRITION_MODEL_PATH, compile=False)

WORK_LABEL_ENCODER = joblib.load(work_label_joblib)
NUTRITION_LABEL_ENCODER = joblib.load(nutrition_label_joblib)


@app.route('/')
def index():
    return 'Hello World'


@app.route('/workout_prediction', methods=['POST']) # JSON Request -> UserId
def predict_workout():

    if request.method == 'POST':
        user_id = request.get_json().get('UserId', None)

        if user_id:
            connection = open_connection()
            
            df_workout = pd.read_json(workout_json, orient='index').drop(WORKOUT_DROP, axis=1)
            df_user = get_user_df(connection, user_id)
            df_hist = get_hist_work_df(connection, user_id)
            
            connection.close()

            df_user['gender'] = df_user['gender'].str.title()
            df_user['level'] = df_user['level'].str.title()
            df_workout['workout_id'] = df_workout.index
            gender_work = df_workout[
                (df_workout.gender == df_user.gender.values[0]) & (~df_workout.workout_id.isin(df_hist[df_hist.user_id == df_user.user_id.values[0]].workout_id))
            ]

            n = 10
            top_n_prediction = work_predict_n(WORK_MODEL, WORK_LABEL_ENCODER, n, gender_work, df_user)
            df_prediction = gender_work.set_index('workout_id').loc[top_n_prediction].reset_index()

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success"
                },
                "data": df_prediction.to_json()
            }), 200
        else:
            return jsonify({
                'status': {
                    'code': 422,
                    'message': 'Unprocessable'
                },
                'data': None,
            }), 422
    else:
        return jsonify({
            'status': {
                'code': 405,
                'message': 'Method not allowed'
            },
            'data': None,
        }), 405


@app.route('/nutrition_prediction', methods=['GET']) # JSON Request -> UserId
def predict_nutrition():

    if request.method == 'GET':
        user_id = '9be2e512-8645-4c8b-b54b-a6823d65dd5a' #fggdsafdsgfsdfadafdsfdsfaafdss

        if user_id:
            connection = open_connection()

            df_user = get_user_bmi_df(connection, user_id).apply(_get_goals_type, axis=1)
            df_user['Age'] = df_user['Age'].apply(_get_age)
            
            connection.close()

            prediction = nutrition_predict(NUTRITION_MODEL, df_user, NUTRITION_LABEL_ENCODER)

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success"
                },
                "data": prediction
            }), 200
        else:
            return jsonify({
                'status': {
                    'code': 422,
                    'message': 'Unprocessable'
                },
                'data': None,
            }), 422
    else:
        return jsonify({
            'status': {
                'code': 405,
                'message': 'Method not allowed'
            },
            'data': None,
        }), 405


def _get_goals_type(df_user):
    goal_type = ['Weight Loss', 'Mild Weight Loss', 'Maintain Weight', 'Mild Weight Gain', 'Gain Weight']
    percent = round(df_user.Goal / df_user.Weight, 2)

    idx = (percent >= 0.95), (percent >= 0.98), (percent >= 1.02), (percent >= 1.05)
    df_user['Goal'] = goal_type[sum(idx)]

    return df_user


def _get_age(birth_date):
    age = (date.today() - birth_date) // timedelta(days=365.2422)

    return age


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
