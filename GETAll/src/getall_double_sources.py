from __future__ import annotations

import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import math

import pandas as pd

from getall_basic import GETAllNetwork

class DoubleSourceGETAllNetwork(GETAllNetwork):


    # this method allows to define dense layers for each source
    def create_dense_entity_layers(self, starting_dim, dropout_value):

        # map of dimension and layer dims
        linear_projection_layers_dims = {    
            4096: [4096, 2048, 1024, 512, 256, 128],
            2048: [2048, 1024, 512, 256, 128],
            1024: [1024, 768, 512, 256, 128],
            768:  [768, 512, 256, 128],
            512:  [512, 384, 256, 128],
            384:  [384, 256, 128],
            256:  [256, 256],
            192:  [192, 128],
            128:  [128, 128],
            100:  [100, 128]
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

        if len(self.modality_features) != 2:
            raise ValueError("This class only supports two sources of features\n"
                             f"The following number of modality features were found: {len(self.modality_features)}")


        self.first_modality_features = torch.nn.Parameter(self.modality_features[0].float(), requires_grad=False)
        self.second_modality_features = torch.nn.Parameter(self.modality_features[1].float(), requires_grad=False)

        self.allowed_dims = [4096, 2048, 1024, 768, 512, 384, 256, 192, 128, 100]

        starting_dim = self.first_modality_features.size(1)
        if starting_dim not in self.allowed_dims:
            raise ValueError(f'Embedding dimensionality not supported: passed {starting_dim},\n'
                f' possible dims:{set(self.allowed_dims.keys())}')

        starting_dim = self.second_modality_features.size(1)
        if starting_dim not in self.allowed_dims:
            raise ValueError(f'Embedding dimensionality not supported: passed {starting_dim},\n'
                f' possible dims:{set(self.allowed_dims.keys())}')


        self.first_dense_user = self.create_dense_entity_layers(self.first_modality_features.size(1), dropout_value)
        self.first_dense_item = self.create_dense_entity_layers(self.first_modality_features.size(1), dropout_value)

        self.second_dense_user = self.create_dense_entity_layers(self.second_modality_features.size(1), dropout_value)
        self.second_dense_item = self.create_dense_entity_layers(self.second_modality_features.size(1), dropout_value)


class GETAllNetworkConcat(DoubleSourceGETAllNetwork):

    def __init__(self, modality_features, dropout_value):
        super().__init__(modality_features, dropout_value)

        
        self.linear_user = torch.nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.linear_item = torch.nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),            
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.compute_score = torch.nn.Sequential(
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

        first_users_features = self.first_modality_features[user_idx]
        first_items_features = self.first_modality_features[item_idx]
        second_users_features = self.second_modality_features[user_idx]
        second_items_features = self.second_modality_features[item_idx]

        x1_user = self.first_dense_user(first_users_features)
        x1_item = self.first_dense_item(first_items_features)

        x2_user = self.second_dense_user(second_users_features)
        x2_item = self.second_dense_item(second_items_features)
        
        concat_user = self.linear_user(torch.cat([x1_user, x2_user], dim=-1))
        concat_item = self.linear_item(torch.cat([x1_item, x2_item], dim=-1))
        out = self.compute_score(torch.cat([concat_user, concat_item], dim=-1))

        return out

class GETAllNetworkSimplifiedConcat(DoubleSourceGETAllNetwork):

    def __init__(self, modality_features, dropout_value):
        super().__init__(modality_features, dropout_value)

        graph_size = self.first_modality_features.size(1)
        if graph_size == 384:

            self.first_dense_user = torch.nn.Sequential(
                nn.Dropout(dropout_value),
                nn.Linear(self.first_modality_features.size(1), 64),
                nn.ReLU()
            )

            self.first_dense_item = torch.nn.Sequential(
                nn.Dropout(dropout_value),
                nn.Linear(self.first_modality_features.size(1), 64),
                nn.ReLU()
            )

        else: # k= 192

            self.first_dense_user = torch.nn.Sequential(
                nn.Dropout(dropout_value),
                nn.Linear(self.first_modality_features.size(1), 64),
                nn.ReLU()
            )

            self.first_dense_item = torch.nn.Sequential(
                nn.Dropout(dropout_value),
                nn.Linear(self.first_modality_features.size(1), 64),
                nn.ReLU()
            )

        self.second_dense_user = torch.nn.Sequential(
            nn.Linear(self.second_modality_features.size(1), 64),
            nn.ReLU()
        )

        self.second_dense_item = torch.nn.Sequential(
            nn.Linear(self.second_modality_features.size(1), 64),
            nn.ReLU()
        )

        self.linear_user = torch.nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            
        )

        self.linear_item = torch.nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        self.compute_score = torch.nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.init_()

    def forward(self, x):
        user_idx = x[0]
        item_idx = x[1]

        first_users_features = self.first_modality_features[user_idx]
        first_items_features = self.first_modality_features[item_idx]
        second_users_features = self.second_modality_features[user_idx]
        second_items_features = self.second_modality_features[item_idx]

        x1_user = self.first_dense_user(first_users_features)
        x1_item = self.first_dense_item(first_items_features)

        x2_user = self.second_dense_user(second_users_features)
        x2_item = self.second_dense_item(second_items_features)

        concat_user = self.linear_user(torch.cat([x1_user, x2_user], dim=-1))
        concat_item = self.linear_item(torch.cat([x1_item, x2_item], dim=-1))
        out = self.compute_score(torch.cat([concat_user, concat_item], dim=-1))

        return out

class GETAllNetworkMerge(DoubleSourceGETAllNetwork):

    def __init__(self, modality_features, dropout_value):
        super().__init__(modality_features, dropout_value)


        self.attention_user = torch.nn.Sequential(
            nn.Linear(256, 128),
            nn.Softmax(dim=-1),
        )

        self.attention_item = torch.nn.Sequential(
            nn.Linear(256, 128),
            nn.Softmax(dim=-1),
        )

        self.attention_cross = torch.nn.Sequential(
            nn.Linear(1, 128),
            nn.Softmax(dim=-1),
        )

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

    @staticmethod
    def merge(x_1, x_2, attention):
        return attention * x_1 + (1 - attention) * x_2

    def forward(self, x):
        user_idx = x[0]
        item_idx = x[1]

        first_users_features = self.first_modality_features[user_idx]
        first_items_features = self.first_modality_features[item_idx]
        second_users_features = self.second_modality_features[user_idx]
        second_items_features = self.second_modality_features[item_idx]

        x1_user = self.first_dense_user(first_users_features)
        x1_item = self.first_dense_item(first_items_features)

        x2_user = self.second_dense_user(second_users_features)
        x2_item = self.second_dense_item(second_items_features)

        concat_user = torch.cat([x1_user, x2_user], dim=-1)
        attention_user = self.attention_user(concat_user)
        merged_user = self.merge(x1_user, x2_user, attention_user)

        concat_item = torch.cat([x1_item, x2_item], dim=-1)
        attention_item = self.attention_item(concat_item)
        merged_item = self.merge(x1_item, x2_item, attention_item)

        attention_weights = self.attention_cross(torch.sum(merged_user * merged_item, dim=-1).unsqueeze(-1))
        merged_item_user = torch.add(merged_user * attention_weights, merged_item * (1 - attention_weights))

        out = self.compute_score(merged_item_user)

        return out






























