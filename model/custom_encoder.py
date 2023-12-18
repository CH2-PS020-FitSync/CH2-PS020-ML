# @title
import json

import numpy as np


class CustomEncoder:
    def __init__(self, encoder_path=None, load_encoder=False):
        self.encoder_path = encoder_path
        self.encoder = self.load_encoder() if encoder_path and load_encoder else dict()

    def load_encoder(self):

        with open(self.encoder_path, 'r') as f:
            return json.load(f)

    def save_encoder(self, save_path):

        with open(save_path, 'w+') as f:
            json.dump(self.encoder, f)

    def fit(self, name, data):
        self.encoder[name] = self.encoder.get(name, dict())

        unique_values = np.unique(data)
        previous_n = len(self.encoder[name])
        new_entries = {}

        # Deal with it for now
        for value in unique_values:
            if isinstance(value, list):
                for key in value:
                    if key not in self.encoder[name] and key not in new_entries.keys():
                        new_entries[key] = previous_n
                        previous_n += 1
            elif value not in self.encoder[name]:
                new_entries[value] = previous_n
                previous_n += 1

        if new_entries:
            self.encoder[name].update(new_entries)

    def transform(self, name, data):
        if name in self.encoder:
            if isinstance(data.values[0], list):
                return [[self.encoder[name].get(key) for key in d] for d in data]

            return data.map(self.encoder[name])
        else:
            raise ValueError(f"No encoder found for '{name}'")

    def fit_transform(self, name, data):
        self.fit(name, data)

        return self.transform(name, data)

    def inverse_transform(self, name, data):
        if name in self.encoder:
            reverse_encoder = {
                val: key
                for key, val in self.encoder[name].items()
            }

            return data.map(reverse_encoder)
        else:
            raise ValueError(f"No encoder found for '{name}'")
