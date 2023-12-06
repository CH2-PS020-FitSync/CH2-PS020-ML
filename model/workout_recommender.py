import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).parent.parent

MODEL_PATH = ROOT / 'model/saved_model/dummy_workout_recommend.h5'
FEATURES = ['gender_x', 'level_x', 'title', 'type', 'body_part', 'gender_y', 'level_y']
LABEL_ENCODER = dict()

workout_json = ROOT / 'data/gymvisual-cleaned-2.json'
user_json = ROOT / 'data/dummy_user.json'
user_act_json = ROOT / 'data/dummy_user_act.json'
hist_json = ROOT / 'data/work-hist.json'
label_joblib = ROOT / 'workout_hist_label.joblib'


def get_col_to_encode(*dataframes, output_path=None):
    cols = set()

    for dataframe in dataframes:
        dataframe_cols = dataframe.select_dtypes(exclude=[np.number])
        cols.update(dataframe_cols)

        for col in dataframe_cols.columns:
            if col != 'name':
                LABEL_ENCODER[col] = LABEL_ENCODER.get(col, LabelEncoder().fit(dataframe[col]))

    if 'name' in cols:
        cols.remove('name')

    if output_path is not None:
        joblib.dump(LABEL_ENCODER, output_path)

    return cols


def encode_hist_work(df_workout, df_hist):
    encoded_df_workout = df_workout.copy()
    encoded_df_hist = df_hist.copy()

    columns_to_encode = get_col_to_encode(encoded_df_workout, encoded_df_hist, output_path=label_joblib) # Inplace encode

    for col in columns_to_encode:

        if col in encoded_df_workout.columns:
            encoded_df_workout[col] = LABEL_ENCODER[col].transform(encoded_df_workout[col])

        if col in encoded_df_hist.columns:
            encoded_df_hist[col] = LABEL_ENCODER[col].transform(encoded_df_hist[col])


    return encoded_df_workout, encoded_df_hist


def train(workout_data, model_path, train=True, history_data=None, user_data=None):
    # if history_data is not None and len(history_data.title.unique()) >= 5:
    merged_data = pd.merge(history_data, workout_data, on='title').dropna()
    X_train, X_test, Y_train, Y_test = train_test_split(merged_data[FEATURES], merged_data['rating'], test_size=0.2)
    # merged_data = merged_data.drop_duplicates(subset=['title'], keep='last')

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='relu'),
    ])

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mse', 'mae']
    )

    history = model.fit(
        X_train, Y_train,
        epochs=100,
        validation_data=(X_test, Y_test),
        verbose=2
    )

    loss = model.evaluate(X_test, Y_test)
    print(f"Test loss: {loss}")

    model.save(model_path)

    return model


def predict_n(model, label_encoder, n, name, gender_workout, df_hist, df_user):
    user = df_user.copy()[df_user.name == name]
    history = df_hist.copy()[df_hist.name == name]

    gender_workout = gender_workout.copy()

    columns_to_encode = get_col_to_encode(user, gender_workout)
    print(columns_to_encode)

    for col in columns_to_encode:

        if col in user.columns:
            user[col] = label_encoder[col].transform(user[col])

        if col in gender_workout.columns:
            print(label_encoder[col].classes_)
            gender_workout[col] = label_encoder[col].transform(gender_workout[col])

    user_merge = pd.merge(gender_workout, user, how='cross')

    result = model.predict(user_merge[FEATURES])

    top_n_index = np.argpartition(-result[:, 0], n)[:n] # Top n max values index
    sorted_top_n_index = top_n_index[np.argsort(-result[top_n_index][:, 0])] # Sorted from max to min

    top_n_recommended = gender_workout.iloc[sorted_top_n_index]
    top_n_recommended_workout = LABEL_ENCODER['title'].inverse_transform(top_n_recommended.title)

    return top_n_recommended_workout


if __name__ == '__main__':
    with open(workout_json, 'r') as f:
        workout_f = json.load(f)

    df_workout = pd.json_normalize(workout_f)
    df_user = pd.read_json(user_json)
    df_hist = pd.read_json(user_act_json)
    # df_hist = pd.read_json(hist_json)

    df_workout.drop(
        ['desc', 'jpg', 'gif', 'duration.desc', 'duration.min', 'duration.rep', 'duration.set', 'duration.sec'],
        axis=1, inplace=True
    )
    
    df_workout_copy, df_hist_copy = encode_hist_work(df_workout, df_hist)

    model = train(df_workout_copy, MODEL_PATH, history_data=df_hist_copy)


    name = 'Thomas Lewis'

    user = df_user[df_user.name == name]
    gender_work = df_workout[
        (df_workout.gender == user.gender.values[0]) & (~df_workout.title.isin(df_hist[df_hist.name == name].title))
    ]
    n = 10

    print(user)

    top_n_prediction = predict_n(model, LABEL_ENCODER, n, name, gender_work, df_hist, user) # For now use `user` as dummy new user as the database is not updated in realtime

    df_prediction = gender_work.set_index('title').loc[top_n_prediction].reset_index()
    print(df_prediction)
