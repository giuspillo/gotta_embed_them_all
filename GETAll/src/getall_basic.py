from __future__ import annotations

import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import math

import pandas as pd

# GETALL
class GETAllNetwork(torch.nn.Module):

    def __init__(self, modality_features, dropout_value):

        super().__init__()
        self.modality_features = modality_features
        self.dropout_value = dropout_value

    def init_(self):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def import_weights(self, weights):

        i = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = torch.nn.Parameter(torch.from_numpy(weights[i]))
                nn.init.zeros_(m.bias)
                i += 1

        return self

    def return_scores(self, user_idx, item_idx):
        with torch.no_grad():
            scores = self((torch.from_numpy(user_idx).int(),
                           torch.from_numpy(item_idx).int())).cpu()

            if len(item_idx) != 1:
                return scores.squeeze()
            else:
                return scores[0]


class SingleSourceGETAllNetwork(GETAllNetwork):

    # this method allows to define dense layers for each source
    def create_dense_entity_layers(self, starting_dim, dropout_value):

        # map of dimension and layer dims
        linear_projection_layers_dims = {    
            4096: [4096, 2048, 1024, 512, 256, 128, 64],
            2048: [2048, 1024, 512, 256, 128, 64],
            1024: [1024, 768, 512, 256, 128, 64],
            768:  [768, 512, 256, 128, 64],
            512:  [512, 384, 256, 128, 64],
            384:  [384, 256, 128, 64],
            192:  [192, 128, 64],
            128:  [128, 64],
            100:  [100, 64]
        }

        # for each dimension, create a linear layer with relu
        layer_list = [torch.nn.Dropout(dropout_value)]
        dimensions = linear_projection_layers_dims[starting_dim]
        for i, _ in enumerate(dimensions):
            layer_list.append(torch.nn.Linear(dimensions[i], dimensions[i+1]))
            layer_list.append(torch.nn.ReLU())
            if i == len(dimensions)-2:
                break

        return nn.Sequential(*layer_list)

    def __init__(self, modality_features, dropout_value):

        super().__init__(modality_features, dropout_value)

        if len(self.modality_features) != 1:
            raise ValueError("This class only supports a single source of features\n"
                             f"The following number of modality features were found: {len(self.modality_features)}")

        self.modality_features = torch.nn.Parameter(self.modality_features[0].float(), requires_grad=False)

        self.allowed_dims = [4096, 2048, 1024, 768, 512, 384, 192, 128, 100]

        starting_dim = self.modality_features.size(1)
        if starting_dim not in self.allowed_dims:
            raise ValueError(f'Embedding dimensionality not supported: passed {starting_dim},\n'
                f' possible dims:{set(self.allowed_dims.keys())}')


        self.dense_user = self.create_dense_entity_layers(self.modality_features.size(1), dropout_value)
        self.dense_item = self.create_dense_entity_layers(self.modality_features.size(1), dropout_value)



class GETAllNetworkBasic(SingleSourceGETAllNetwork):

    def __init__(self, modality_features, dropout_value):
        super().__init__(modality_features, dropout_value)


        self.compute_score = torch.nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.init_()

    def forward(self, x):

        user_idx = x[0]
        item_idx = x[1]

        users_features = self.modality_features[user_idx]
        items_features = self.modality_features[item_idx]

        dense_users = self.dense_user(users_features)
        dense_items = self.dense_item(items_features)

        out = self.compute_score(torch.cat([dense_users, dense_items], dim=-1))

        return out
