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
        previous = len(self.encoder[name])
        new_entries = {
            val: previous + idx
            for idx, val in enumerate(unique_values)
            if val not in self.encoder[name]
        }

        if new_entries:
            self.encoder[name].update(new_entries)

    def transform(self, name, data):
        if name in self.encoder:
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
