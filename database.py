import os

import pymysql
from flask import jsonify


db_user = os.environ.get('FITSYNC_USER')
db_password = os.environ.get('FITSYNC_PASS')
db_name = 'main_api'
db_connection_name = os.environ.get('FITSYNC_CONN')


def open_connection():
    unix_socket = '/cloudsql/{}'.format(db_connection_name)

    try:
        if os.environ.get('GAE_ENV') == 'standard':
            conn = pymysql.connect( # Retrieving from outside now
                host='35.226.213.127',
                port=3306,
                # unix_socket=unix_socket,
                user=db_user,
                password=db_password,
                db=db_name,
                cursorclass=pymysql.cursors.DictCursor
            )
            
        return conn

    except pymysql.MySQLError as e:
        return e


def get_user(user_id):
    conn = open_connection()

    with conn.cursor() as cursor:
        result = cursor.execute(
            """
                SELECT *
                FROM users
                LIMIT 5
            """
            # WHERE ID = %d (user_id, )
        )
        users = cursor.fetchall()

    conn.close()

    return jsonify(users)

if __name__ == '__main__':
    print(get_user(1))