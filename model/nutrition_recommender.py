from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

try:
    from .custom_encoder import CustomEncoder
except ImportError:
    from custom_encoder import CustomEncoder


ROOT = Path(__file__).parent.parent

MODEL_PATH = ROOT / 'model/saved_model/nutrition_recommend.h5'
FEATURES = ['Age', 'Weight', 'Gender', 'Height', 'Activity_Level', 'Goal']
TARGET = ['Estimated_Calories', 'Estimated_Carbohydrates', 'Estimated_Protein_Mean', 'Estimated_Fat']
CAT_COLS = ['Gender', 'Activity_Level', 'Goal']
LABEL_ENCODER = CustomEncoder()

nutrition_json = ROOT / 'data/nutrition_data.json'
label_json = ROOT / 'nutrition_label.json'


def preprocess_df(df):
    mean_protein = (df.Estimated_Protein_Min + df.Estimated_Protein_Max) / 2.

    df['Activity_Level'].replace({
            'Very Active|Extra Active': 'expert',
            'Moderate|Active': 'intermediate',
            'Sedentary|Light': 'beginner'
        },
        regex=True,
        inplace=True
    )

    df.insert(8, 'Estimated_Protein_Mean', mean_protein, allow_duplicates=True)

    df.drop(
        ['Estimated_Protein_Min', 'Estimated_Protein_Max'],
        axis=1,
        inplace=True
    )


def encode_cols(df, cols, le, output_path):
    for col in cols:
        df[col] = le.fit_transform(col, df[col])

    le.save_encoder(output_path)


def train(dataframe, model_path):
    X_train, X_test, y_train, y_test = \
        train_test_split(dataframe[FEATURES], dataframe[TARGET], train_size=0.9)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(len(FEATURES),)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(len(TARGET))
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mae', 'mse']
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=1000,
        epochs=100,
        verbose=2
    )

    prediction = model.predict(X_test)
    loss = model.evaluate(X_test, y_test)

    print('Prediction:', prediction[:10])
    print('Loss:', loss[:10])
    model.save(model_path)

    return history, model


def nutrition_predict(model, df_user, le):
    df_user['Gender'].replace({
            '(em)?ale': ''
        },
        regex=True,
        inplace=True
    )

    for col in CAT_COLS:
        df_user[col] = le.transform(col, df_user[col])

    X_new = df_user[FEATURES]
    prediction = model.predict(X_new)
    result = {i: str(k) for i, k in zip(TARGET, prediction[0])}

    return result


if __name__ == '__main__':
    df_nutrition = pd.read_json(nutrition_json)

    preprocess_df(df_nutrition)
    encode_cols(df_nutrition, CAT_COLS, LABEL_ENCODER, label_json)

    history, model = train(df_nutrition, MODEL_PATH)


    # Weight goals must be transformed from actual goal in kgs to percentage of body mass to lose or gain
    new_user = {
        'Age': 27,
        'Weight': 65,
        'Gender': 'f',
        'Height': 164,
        'Activity_Level': 'expert',
        'Goal': 'Weight Loss'
    }
    df_user = pd.DataFrame([new_user])

    prediction = nutrition_predict(model, df_user, LABEL_ENCODER)

    print('Predicted Nutritional Needs:')
    for target, pred in zip(TARGET, prediction.keys()):
        print(f'{target:<25}:{prediction[pred]}')


    # tf.keras.utils.plot_model(model, to_file=ROOT/'model/nutrition_reg.png', show_shapes=True)
    
    
    # import matplotlib.pyplot as plt

    # acc = history.history['mse']
    # val_acc = history.history['val_mse']
    # loss = history.history['loss']
    # val_loss = history.history['val_mse']

    # epochs = range(100)

    # plt.plot(epochs, acc, 'r', label='Training MSE')
    # plt.plot(epochs, val_acc, 'b', label='Validation MSE')
    # plt.title('Nutrition models error')
    # plt.legend(loc=0)
    
    # plt.savefig(ROOT/'model/nutrition_error.png')
