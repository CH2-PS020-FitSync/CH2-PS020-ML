import os
from datetime import date, timedelta

import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, request

from database import (get_hist_work_df, get_user_bmi_df, get_user_df,
                      open_connection)
from model.custom_encoder import CustomEncoder
from model.nutrition_recommender import MODEL_PATH as NUTRITION_MODEL_PATH
from model.nutrition_recommender import label_json as nutrition_label_json
from model.nutrition_recommender import nutrition_predict
from model.workout_recommender_embedding import MODEL_PATH as WORK_MODEL_PATH
from model.workout_recommender_embedding import WORKOUT_DROP
from model.workout_recommender_embedding import label_json as work_label_json
from model.workout_recommender_embedding import work_predict_n, workout_json

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


app = Flask(__name__)

HOST = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
PORT = int(os.environ.get('FLASK_RUN_PORT', 8080))

WORK_MODEL = tf.keras.models.load_model(WORK_MODEL_PATH, compile=False)
NUTRITION_MODEL = tf.keras.models.load_model(NUTRITION_MODEL_PATH, compile=False)

WORK_LABEL_ENCODER = CustomEncoder(encoder_path=work_label_json, load_encoder=True)
NUTRITION_LABEL_ENCODER = CustomEncoder(encoder_path=nutrition_label_json, load_encoder=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    return jsonify({
        'about': 'FitSync Inference API',
        'endpoints': [
            '/workout_prediction',
            '/nutrition_prediction'
        ],
        'authors': {
            'machine_learning': [
                { 
                    'name': 'Steven Tribethran',
                    'bangkit_id': 'M694BSY0582',
                    'github_profile': 'https://www.github.com/Insisted'
                },
                { 
                    'name': 'Darrel Cyril Gunawan',
                    'bangkit_id': 'M108BSY1617',
                    'github_profile': 'https://www.github.com/Darrelcyril29'
                },
                { 
                    'name': 'Nigel Kusdenata',
                    'bangkit_id': 'M108BSY1102',
                    'github_profile': 'https://www.github.com/NigelKus'
                }
            ],
            'cloud_computing': [
                { 
                    'name': 'Muhammad Alfayed Dennita',
                    'bangkit_id': 'C134BSY3479',
                    'github_profile': 'https://www.github.com/AlfayedDennita'
                },
                { 
                    'name': 'Alida Shidqiya Naifa Ulmuflikhun',
                    'bangkit_id': 'C248BSX4205',
                    'github_profile': 'https://www.github.com/alidasn'
                }
            ],
            'mobile_development': [
                { 
                    'name': 'Vincent',
                    'bangkit_id': 'A694BSY2946',
                    'github_profile': 'https://www.github.com/Vincent-2125250004'
                },
                { 
                    'name': 'Hidayat',
                    'bangkit_id': 'A550BKY4421',
                    'github_profile': 'https://www.github.com/hkvil'
                }
            ]
        }
    })


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

            if df_user.empty:
                return jsonify({
                    'data': None,
                    'status': {
                        'code': 404,
                        'message': 'User not found'
                    }
                }), 404

            df_user['gender'] = df_user['gender'].str.title()
            df_user['level'] = df_user['level'].str.title()
            df_workout['workout_id'] = df_workout.index
            df_workout['bodyPart'] = df_workout['bodyPart'].str.split(', ')
            gender_work = df_workout[
                (df_workout.gender == df_user.gender.values[0]) & \
                    (~df_workout.workout_id.isin(df_hist[df_hist.user_id == df_user.user_id.values[0]].workout_id))
            ]

            n = 10
            top_n_prediction = work_predict_n(WORK_MODEL, WORK_LABEL_ENCODER, n, gender_work, df_user)
            prediction_id = top_n_prediction.index.values.tolist()

            return jsonify({
                'data': prediction_id,
                'status': {
                    'code': 200,
                    'message': 'Success'
                }
            }), 200
        else:
            return jsonify({
                'data': None,
                'status': {
                    'code': 422,
                    'message': 'Unprocessable'
                }
            }), 422
    else:
        return jsonify({
            'data': None,
            'status': {
                'code': 405,
                'message': 'Method not allowed'
            }
        }), 405


@app.route('/nutrition_prediction', methods=['POST']) # JSON Request -> UserId
def predict_nutrition():

    if request.method == 'POST':
        user_id = request.get_json().get('UserId', None)

        if user_id:
            connection = open_connection()

            df_user = get_user_bmi_df(connection, user_id).apply(_get_goals_type, axis=1)
            
            connection.close()

            if df_user.empty:
                return jsonify({
                    'data': None,
                    'status': {
                        'code': 404,
                        'message': 'User not found'
                    }
                }), 404
            
            df_user = df_user.drop_duplicates(subset=['UserId'], keep='last')
            df_user['Age'] = df_user['Age'].apply(_get_age)

            prediction = nutrition_predict(NUTRITION_MODEL, df_user, NUTRITION_LABEL_ENCODER)

            return jsonify({
                'data': prediction,
                'status': {
                    'code': 200,
                    'message': 'Success'
                }
            }), 200
        else:
            return jsonify({
                'data': None,
                'status': {
                    'code': 422,
                    'message': 'Unprocessable'
                }
            }), 422
    else:
        return jsonify({
            'data': None,
            'status': {
                'code': 405,
                'message': 'Method not allowed'
            }
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
    app.run(host=HOST, port=PORT)
