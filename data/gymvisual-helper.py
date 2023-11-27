import json
import re

import pandas as pd


def clean(df, gym):
    equip = df.to_numpy()
    temp = []

    print(len(equip), len(gym))

    for i, dataf in zip(gym, equip):
        a = re.sub(r'.*({.+}).*', r'\1', i['result'])
        b = re.sub(r'\s{2,}|\n', '', a)
        c = re.sub('\"title\": \"{(.+)}', '\"title\": \"\\1', b)
        res = eval(re.sub(r'Q:.+', '', c))

        try:
            temp.append({
                'title': dataf[0],
                'type': dataf[1],
                'body_part': dataf[2],
                'gender': dataf[3],
                'desc': res['desc'],
                'level': res['level'],
                'duration': parse_time(res['time'])
            })
        except: print(res)

    return temp


def parse_time(time):
    parsed = re.findall(r'([0-9-]+)\s*(\w+)', time)
    res = dict()
    res['desc'] = time

    for i in parsed:
        unit = i[1][:3]

        if re.match(r'\d+\s*sets*\s*of\s*\d+\s*(min|sec)\w+', time) and unit in ['sec', 'min']:
            res[unit] = str(int(i[0]) * int(res['set']))
        else:
            res[unit] = i[0]

    return res


def get_vis(clean, vis):
    clean = pd.DataFrame(clean)
    vis = pd.DataFrame(vis)
    merged = pd.merge(clean, vis, on='title').drop_duplicates(subset=['title'], keep='first')

    print(len(clean), len(vis), len(merged))

    merged.to_json('./ML/data/gymvisual-cleaned-2.json', orient='records')


if __name__ == '__main__':
    df = pd.read_json('./ML/data/gymvisual-cleaned.json')

    with open('./ML/data/gymvisual-unclean-2.json', 'r') as f:
        uncleaned_gym = json.load(f)

    with open('./ML/data/gymvisual-image_urls.json', 'r') as f:
        vis = json.load(f)

    cleaned = clean(df, uncleaned_gym)

    get_vis(cleaned, vis)