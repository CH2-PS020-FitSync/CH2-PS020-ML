import os

import pymysql
from flask import jsonify


db_user = os.environ.get('FITSYNC_USER')
db_password = os.environ.get('FITSYNC_PASS')
db_name = 'main_api'
db_connection_name = 'fitsync-406408:us-central1:main-mysql'


def open_connection():
    unix_socket = '/cloudsql/{}'.format(db_connection_name)

    try:
        if os.environ.get('GAE_ENV') == 'standard':
            conn = pymysql.connect(
                user=db_user,
                password=db_password,
                unix_socket=unix_socket,
                db=db_name,
                cursorclass=pymysql.cursors.DictCursor
            )

    except pymysql.MySQLError as e:
        return e

    return conn


def get_user(user_id):
    conn = open_connection()

    with conn.cursor() as cursor:
        result = cursor.execute(
            """
                SELECT *
                FROM users
                WHERE id = %d
            """,
            (user_id, )
        )
        users = cursor.fetchall()

        return jsonify(users) or None
