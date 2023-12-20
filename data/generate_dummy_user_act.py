# Dummy User

import json
from pathlib import Path

import pandas as pd
from faker import Faker
from numpy.random import choice, randint, random


ROOT = Path(__file__).parent

GENDER = ['Male', 'Female']
LEVEL = ['Beginner', 'Intermediate', 'Expert']
WORKOUT_JSON = ROOT / 'data/gymvisual-use-model.json'
OUTPUT_USER = ROOT / 'data/dummy_user.json'
OUTPUT_ACT = ROOT / 'data/dummy_user_act.json'

faker = Faker()


def generate_user():
    df_user = [
        {
            'user_id': f'dummy_{id}',
            'name': faker.name(),
            'gender': choice(GENDER),
            'weight': round(random(), 1) + randint(40, 70),
            'height': randint(150, 180),
            'age': randint(15, 30),
            'level': choice(LEVEL)
        } for id in range(100)
    ]

    with open(OUTPUT_USER, 'w+') as f:
        json.dump(df_user, f)

    return df_user


def generate_act():
    df_user = pd.DataFrame(generate_user())

    df_hist = []
    df_workout = pd.read_json(WORKOUT_JSON, orient='index')

    df_workout.drop(
        df_workout[df_workout.level == 'Beginner'].sample(frac=.8).index,
        inplace=True
    ) # Just to scale down the scope

    for name in df_user.name:
        user = df_user[df_user.name == name]
        u_level = LEVEL.index(user.level.values[0])
        u_gender = user.gender.values[0]

        for _ in range(randint(20, 100)):
            workout_det_level = df_workout[(df_workout.gender == u_gender) & (random() < 0.4 or df_workout.level == user.level.values[0])]
            workout = workout_det_level.sample(1)
            w_level = LEVEL.index(workout.level.values[0])
            diff = abs(u_level - w_level)
            rating = max(0, randint(5, 10) - (randint(3, 6) if diff > 1 else randint(2, 4) if diff else randint(0, 1)))

            df_hist.append(
                {
                    'user_id': user.user_id.values[0],
                    'name': user.name.values[0],
                    'gender': u_gender,
                    'exercise_id': workout.index.values[0],
                    'level': user.level.values[0],
                    'rating': rating,
                    'diff': diff
                }
            )

    with open(OUTPUT_ACT, 'w+') as f:
        json.dump(df_hist, f)

if __name__ == '__main__':
    generate_act()
