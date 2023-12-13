import numpy as np
import pandas as pd

df_path = './data/gymvisual-cleaned-2.json'
df = pd.read_json(df_path)

df.dropna(inplace=True)

scope = df.loc[df.level == 'Beginner']
interest = scope.value_counts('body_part').nlargest(20)


for i in interest.index:
    part = scope.loc[(scope.body_part == i)]

    for j in ('Male', 'Female'):
        part_gender = part.loc[part.gender == j]
        count = part_gender.shape[0]
        remove_n = max(0, count - 15)

        drop_indices = np.random.choice(
            part_gender.index, remove_n, replace=False
        )
        
        df.drop(drop_indices, inplace=True)


df.to_json('./data/gymvisual-use-db.json', orient='records')
