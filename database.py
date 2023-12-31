import os
from datetime import date, timedelta

import pandas as pd
import pymysql
from flask import jsonify
from google.cloud.sql.connector import Connector, IPTypes


connector = Connector()

db_user = os.environ.get('FITSYNC_USER')
db_password = os.environ.get('FITSYNC_PASS')
db_name = 'main_api'
db_connection_name = os.environ.get('FITSYNC_CONN')


def open_connection():
    # if os.environ.get('GAE_ENV') == 'standard':
    #     conn = pymysql.connect(
    #         unix_socket=unix_socket,
    #         user=db_user,
    #         password=db_password,
    #         db=db_name,
    #         cursorclass=pymysql.cursors.DictCursor
    #     )
        
    connection = connector.connect(
        db_connection_name,
        'pymysql',
        user=db_user,
        password=db_password,
        db=db_name,
        ip_type=IPTypes.PUBLIC
    )

    return connection


def get_user_df(connection, user_id):
    result = pd.read_sql(
        """
            SELECT
                id as user_id,
                gender,
                level
            FROM Users
            WHERE id = %(user_id)s;
        """,
        connection,
        params={
            'user_id': user_id
        }
    )

    return result


def get_user_bmi_df(connection, user_id): # 'Age', 'Weight', 'Gender', 'Height', 'Activity_Level', 'Goal'
    result = pd.read_sql(
        """
            SELECT
                U.id as UserId,
                U.birthDate as Age,
                U.gender as Gender,
                B.weight as Weight,
                B.height as Height,
                U.level as Activity_Level,
                U.goalWeight as Goal
            FROM 
                Users U
            LEFT OUTER JOIN BMIs B
                ON U.id = B.UserId
            WHERE 
                U.id = %(user_id)s
        """,
        connection,
        params={
            'user_id': user_id
        }
    )

    return result


def get_hist_work_df(connection, user_id):
    today = date.today()
    month_ago = today - timedelta(days=30) # Just assume it's 30 days

    result = pd.read_sql(
        """
            SELECT
                ExerciseId as workout_id,
                UserId as user_id,
                rating
            FROM Workouts
            WHERE userid = %(user_id)s AND createdAt BETWEEN %(date_ago)s AND %(date_now)s;
        """,
        connection,
        params={
            'user_id': user_id,
            'date_ago': today.strftime('%Y/%m/%d'),
            'date_now': month_ago.strftime('%Y/%m/%d')
        }
    )

    return result
