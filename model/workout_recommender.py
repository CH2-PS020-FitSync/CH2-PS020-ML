import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

try:
    from .custom_encoder import CustomEncoder
except ImportError:
    from custom_encoder import CustomEncoder


ROOT = Path(__file__).parent.parent

MODEL_PATH = ROOT / 'model/saved_model/dummy_workout_recommend.h5'
FEATURES = ['user_id', 'gender_x', 'level_x', 'workout_id', 'type', 'bodyPart', 'gender_y', 'level_y']
WORKOUT_DROP = ['desc', 'jpg', 'gif', 'duration', '__collections__']
LABEL_ENCODER = CustomEncoder()

workout_json = ROOT / 'data/gymvisual-use-model.json'
user_json = ROOT / 'data/dummy_user.json'
user_act_json = ROOT / 'data/dummy_user_act.json'
hist_json = ROOT / 'data/work-hist.json'
label_json = ROOT / 'workout_hist_label.json'


def get_col_to_encode(*dataframes, le=None, output_path=None):
    cols = set()
    not_encoded = {'name'}

    for dataframe in dataframes:
        dataframe_cols = dataframe.select_dtypes(exclude=[np.number])
        cols.update(dataframe_cols)

        if le is None:
            continue

        for col in dataframe_cols.columns:
            if col not in not_encoded:
                le.fit(col, dataframe[col])

    cols = cols.symmetric_difference(not_encoded)

    if output_path is not None:
        le.save_encoder(output_path)

    return cols


def encode_hist_work(df_workout, df_hist, le=dict(), output_path='.'):
    encoded_df_workout = df_workout.copy()
    encoded_df_hist = df_hist.copy()

    columns_to_encode = \
        get_col_to_encode(encoded_df_workout, encoded_df_hist, le=le, output_path=output_path) # Inplace encode

    for col in columns_to_encode:

        if col in encoded_df_workout.columns:
            encoded_df_workout[col] = le.transform(col, encoded_df_workout[col])

        if col in encoded_df_hist.columns:
            encoded_df_hist[col] = le.transform(col, encoded_df_hist[col])


    return encoded_df_workout, encoded_df_hist


def train(workout_data, model_path, train=True, history_data=None, user_data=None):
    # if history_data is not None and len(history_data.id.unique()) >= 5:
    merged_data = pd.merge(history_data, workout_data, on='workout_id').dropna()
    X_train, X_test, Y_train, Y_test = \
        train_test_split(merged_data[FEATURES], merged_data['rating'], test_size=0.2)
    # merged_data = merged_data.drop_duplicates(subset=['id'], keep='last')

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
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

    print(f'Test loss: {loss}')
    model.save(model_path)

    return model


def work_predict_n(model, le, n, gender_workout, df_user):
    user = df_user.copy()
    gender_workout = gender_workout.copy()

    columns_to_encode = get_col_to_encode(user, gender_workout)

    for col in columns_to_encode:

        if col in user.columns:
            user[col] = le.transform(col, user[col])

        if col in gender_workout.columns:
            gender_workout[col] = le.transform(col, gender_workout[col])

    user_merge = pd.merge(gender_workout, user, how='cross')

    result = model.predict(user_merge[FEATURES])

    top_n_index = np.argpartition(-result[:, 0], n)[:n] # Top n max values index
    sorted_top_n_index = top_n_index[np.argsort(-result[top_n_index][:, 0])] # Sorted from max to min

    top_n_recommended = gender_workout.iloc[sorted_top_n_index]
    top_n_recommended_workout = le.inverse_transform('workout_id', top_n_recommended.workout_id)

    return top_n_recommended_workout


if __name__ == '__main__':
    df_workout = pd.read_json(workout_json, orient='index')
    df_user = pd.read_json(user_json)
    df_hist = pd.read_json(user_act_json)
    # df_hist = pd.read_json(hist_json)

    df_workout['workout_id'] = df_workout.index
    df_workout.drop(
        WORKOUT_DROP,
        axis=1, inplace=True
    )
    df_hist.rename(columns={'exercise_id': 'workout_id'}, inplace=True)
    
    df_workout_copy, df_hist_copy = \
        encode_hist_work(df_workout, df_hist, LABEL_ENCODER, label_json)

    model = train(df_workout_copy, MODEL_PATH, history_data=df_hist_copy)


    user = pd.DataFrame([{
        "user_id": "x",
        "name": "New",
        "gender": "Female",
        "weight": 62.5,
        "height": 155,
        "age": 17,
        "level": "Expert"
    }])

    gender_work = df_workout[
        (df_workout.gender == user.gender.values[0]) & \
            (~df_workout.workout_id.isin(df_hist[df_hist.user_id == user.user_id.values[0]].workout_id))
    ]
    n = 10

    print(user)

    top_n_prediction = work_predict_n(model, LABEL_ENCODER, n, gender_work, user)

    df_prediction = gender_work.set_index('workout_id').loc[top_n_prediction].reset_index()
    print(df_prediction)
