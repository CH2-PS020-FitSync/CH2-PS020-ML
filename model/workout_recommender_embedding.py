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

MODEL_PATH = ROOT / 'model/saved_model/embedding_workout_recommend.h5'
FEATURES = ['user_id', 'gender_x', 'level_x', 'workout_id', 'type', 'bodyPart', 'gender_y', 'level_y']
WORKOUT_DROP = ['desc', 'jpg', 'gif', 'duration', '__collections__']
LABEL_ENCODER = CustomEncoder()
FEATURES_CONFIG = {
    'user_id': {'entity': 'user', 'dtype': tf.int64},
    'gender_x': {'entity': 'user', 'dtype': tf.int64},
    'level_x': {'entity': 'user', 'dtype': tf.int64},
    'workout_id': {'entity': 'workout', 'dtype': tf.int64},
    # 'bodyPart': {'entity': 'workout', 'dtype': tf.int64},
    'gender_y': {'entity': 'workout', 'dtype': tf.int64},
    'level_y': {'entity': 'workout', 'dtype': tf.int64},
    'type': {'entity': 'workout', 'dtype': tf.int64},
}

workout_json = ROOT / 'data/gymvisual-use-model.json'
user_json = ROOT / 'data/dummy_user.json'
user_act_json = ROOT / 'data/dummy_user_act.json'
hist_json = ROOT / 'data/work-hist.json'
label_json = ROOT / 'model/workout_hist_label.json'


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


def train(workout_data, model_path, history_data=None):
    merged_data = pd.merge(history_data, workout_data, on='workout_id').dropna()
    X_train, X_test, Y_train, Y_test = \
        train_test_split(merged_data[FEATURES], merged_data['rating'], test_size=0.2)
    print(merged_data)


    all_workout_features = merged_data['bodyPart'].explode().unique() # Spread the bodyPart
    workout_features_input = tf.keras.layers.Input(shape=(None,), name='bodyPart')

    workout_features_embeddings = tf.keras.layers.Embedding(
        input_dim=len(all_workout_features) + 1,
        output_dim=64
    )(workout_features_input)
    workout_features_embedding = tf.keras.layers.GlobalAveragePooling1D(
        keepdims=True
    )(workout_features_embeddings)

    for name, config in FEATURES_CONFIG.items():
        config['encoding_layer_class'] = tf.keras.layers.IntegerLookup
        config['vocab'] = X_train[name].unique()


    inputs = {
        name: tf.keras.layers.Input(shape=(1,), name=name, dtype=config['dtype'])
        for name, config in FEATURES_CONFIG.items()
    }

    inputs_encoded = {
        name: config['encoding_layer_class'](vocabulary=config['vocab'])(inputs[name])
        for name, config in FEATURES_CONFIG.items()
    }

    embeddings = {
        name: tf.keras.layers.Embedding(
            input_dim=len(config['vocab']) + 1,
            output_dim=64,
            embeddings_regularizer=tf.keras.regularizers.l2(0.1)
        )(inputs_encoded[name])
        for name, config in FEATURES_CONFIG.items()
    }

    # https://stackoverflow.com/questions/49164230/
    # deep-neural-network-skip-connection-implemented-as-summation-vs-concatenation/49179305#49179305
    user_embedding = tf.keras.layers.Concatenate(axis=1)(
        [ 
            embeddings[name]
            for name, config in FEATURES_CONFIG.items()
            if config['entity'] == 'user'
        ]
    )

    workout_embedding = tf.keras.layers.Concatenate(axis=1)(
        [
            embeddings[name]
            for name, config in FEATURES_CONFIG.items()
            if config['entity'] == 'workout'
        ] + [workout_features_embedding]
    )


    dot = tf.keras.layers.Dot(axes=2)([user_embedding, workout_embedding])
    flatten = tf.keras.layers.Flatten()(dot)
    dense_1 = tf.keras.layers.Dense(32, activation='sigmoid')(flatten)
    dense_2 = tf.keras.layers.Dense(1, activation='sigmoid')(dense_1)
    range_output = tf.keras.layers.Lambda(lambda x: 10 * x)(dense_2)


    model = tf.keras.Model(
        inputs=[inputs[name] for name in FEATURES_CONFIG.keys()] + [workout_features_input],
        outputs=range_output
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mse', 'mae']
    )


    X_training_tf = {
        **{name: X_train[name].values for name in FEATURES_CONFIG.keys()},
        'bodyPart': tf.ragged.constant(X_train['bodyPart'].values)
    }
    X_testing_tf = {
        **{name: X_test[name].values for name in FEATURES_CONFIG.keys()},
        'bodyPart': tf.ragged.constant(X_test['bodyPart'].values)
    }


    history = model.fit(
        x=X_training_tf,
        y=Y_train.values,
        epochs=100,
        batch_size=1000,
        validation_data=(X_testing_tf, Y_test.values),
        verbose=2
    )

    loss = model.evaluate(X_testing_tf, Y_test.values)

    print(f'Test loss: {loss}')
    model.save(model_path)

    return history, model


def work_predict_n(model, le, n, gender_workout, df_user):
    user = df_user.copy()
    gender_workout = gender_workout.copy()

    columns_to_encode = get_col_to_encode(user, gender_workout)

    for col in columns_to_encode:

        if col in user.columns:
            user[col] = le.transform(col, user[col])

        if col in gender_workout.columns:
            gender_workout[col] = le.transform(col, gender_workout[col])

    user_merge = pd.merge(gender_workout, user, how='cross')[FEATURES]
    user_merge['user_id'] = user_merge['user_id'].fillna(-1).astype(np.int64) # New User ID

    data = {
        **{name: user_merge[name].values for name in FEATURES_CONFIG.keys()},
        'bodyPart': tf.ragged.constant(user_merge['bodyPart'].values)
    }

    result = model.predict(data)

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
    df_workout['bodyPart'] = df_workout['bodyPart'].str.split(', ')
    df_workout.drop(
        WORKOUT_DROP,
        axis=1, inplace=True
    )
    df_hist.rename(columns={'exercise_id': 'workout_id'}, inplace=True)
    
    df_workout_copy, df_hist_copy = \
        encode_hist_work(df_workout, df_hist, LABEL_ENCODER, label_json)

    history, model = train(df_workout_copy, MODEL_PATH, history_data=df_hist_copy)


    user = pd.DataFrame([{
        'user_id': 'x',
        'name': 'New',
        'gender': 'Female',
        'weight': 62.5,
        'height': 155,
        'age': 17,
        'level': 'Intermediate'
    }])

    gender_work = df_workout[
        (df_workout.gender == user.gender.values[0]) & \
            (~df_workout.workout_id.isin(df_hist[df_hist.user_id == user.user_id.values[0]].workout_id))
    ]
    n = 10

    print(user)

    top_n_prediction = work_predict_n(model, LABEL_ENCODER, n, gender_work, user)

    df_prediction = gender_work[gender_work['workout_id'].isin(top_n_prediction)]
    print(df_prediction)


    # tf.keras.utils.plot_model(model, to_file=ROOT/'model/embedding_workout.png', show_shapes=True)


    # import matplotlib.pyplot as plt
    
    # mse = history.history['mse']
    # val_mse = history.history['val_mse']
    # mae = history.history['mae']
    # val_mae = history.history['val_mae']

    # epochs = range(100)

    # figure, axis = plt.subplots(1, 2, figsize=(16, 9))

    # axis[0].plot(epochs, mse, 'r', label='Training MSE')
    # axis[0].plot(epochs, val_mse, 'b', label='Validation MSE')
    # axis[0].set_title('Workout embedding MSE')
    # axis[0].legend(loc=0)

    # axis[1].plot(epochs, mae, 'r', label='Training MAE')
    # axis[1].plot(epochs, val_mae, 'b', label='Validation MAE')
    # axis[1].set_title('Workout embedding MAE')
    # axis[1].legend(loc=0)

    # plt.savefig(ROOT/'model/workout_embedding_error.png')
