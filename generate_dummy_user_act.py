# Dummy User

from faker import Faker
from numpy.random import choice, random, randint
import pandas as pd
import json
from pathlib import Path

ROOT = Path(__file__).parent

GENDER = ['Male', 'Female']
LEVEL = ['Beginner', 'Intermediate', 'Expert']
WORKOUT_JSON = ROOT / 'data/gymvisual-cleaned-2.json'
OUTPUT_USER = ROOT / 'data/dummy_user.json'
OUTPUT_ACT = ROOT / 'data/dummy_user_act.json'

faker = Faker()


def generate_user():
    df_user = [
        {
            'name': faker.name(),
            'gender': choice(GENDER),
            'weight': round(random(), 1) + randint(40, 70),
            'height': randint(150, 180),
            'age': randint(15, 30),
            'level': choice(LEVEL)
        } for _ in range(100)
    ]

    with open(OUTPUT_USER, 'w+') as f:
        json.dump(df_user, f)

    return df_user


def generate_act():
    df_user = pd.DataFrame(generate_user())
    
    with open(WORKOUT_JSON, 'r') as f:
        workout_f = json.load(f)

    df_hist = []
    df_workout = pd.json_normalize(workout_f)

    df_workout.drop(
        df_workout[df_workout.level == 'Beginner'].sample(frac=.8).index,
        inplace=True
    ) # Just to scale down the scope

    for name in df_user.name:
        user = df_user[df_user.name == name]
        u_level = LEVEL.index(user.level.values[0])
        u_gender = user.gender.values[0]

        for _ in range(randint(20, 100)):
            workout_det_level = df_workout[(df_workout.gender == 'Female') & (random() < 0.4 or df_workout.level == 'Expert')]
            workout = workout_det_level.sample(1)
            w_level = LEVEL.index(workout.level.values[0])
            diff = abs(u_level - w_level)
            rating = max(0, randint(5, 10) - (randint(3, 6) if diff > 1 else randint(2, 4) if diff else randint(0, 1)))

            df_hist.append(
                {
                    'name': user.name.values[0],
                    'gender': user.gender.values[0],
                    'title': workout.title.values[0],
                    'level': user.level.values[0],
                    'rating': rating,
                    'diff': diff
                }
            )

    with open(OUTPUT_ACT, 'w+') as f:
        json.dump(df_hist, f)

if __name__ == '__main__':
    generate_act()