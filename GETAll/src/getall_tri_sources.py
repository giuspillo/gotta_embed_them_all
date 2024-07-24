from __future__ import annotations

import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import math

import pandas as pd

from getall_basic import GETAllNetwork

class ThreeSourceGETAllNetwork(GETAllNetwork):

    # this method allows to define dense layers for each source
    def create_dense_entity_layers(self, starting_dim, dropout_value):

        # map of dimension and layer dims
        linear_projection_layers_dims = {    
            4096: [4096, 2048, 1024, 512, 256],
            2048: [2048, 1024, 512, 256],
            1024: [1024, 768, 512, 256],
            1000: [1000, 768, 512, 256],
            768:  [768, 512, 256],
            512:  [512, 384, 256],
            384:  [384, 256],
            256:  [256, 256],
            192:  [192, 256],
            128:  [128, 256],
            100:  [100, 192, 256]
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

        if len(self.modality_features) != 3:
            raise ValueError("This class only supports three sources of features\n"
                             f"The following number of modality features were found: {len(self.modality_features)}")


        self.first_modality_features = torch.nn.Parameter(self.modality_features[0].float(), requires_grad=False)
        self.second_modality_features = torch.nn.Parameter(self.modality_features[1].float(), requires_grad=False)
        self.third_modality_features = torch.nn.Parameter(self.modality_features[2].float(), requires_grad=False)

        self.allowed_dims = [4096, 2048, 1024, 768, 512, 384, 192, 128, 100]
        allowed_dims = [4096, 2048, 1024, 1000, 768, 512, 384, 256, 192, 128, 100]

        starting_dim = self.first_modality_features.size(1)
        if starting_dim not in allowed_dims:
            raise ValueError(f'Embedding dimensionality not supported: passed {starting_dim},\n'
                f' possible dims:{set(self.allowed_dims.keys())}')

        starting_dim = self.second_modality_features.size(1)
        if starting_dim not in allowed_dims:
            raise ValueError(f'Embedding dimensionality not supported: passed {starting_dim},\n'
                f' possible dims:{set(self.allowed_dims.keys())}')

        starting_dim = self.third_modality_features.size(1)
        if starting_dim not in allowed_dims:
            raise ValueError(f'Embedding dimensionality not supported: passed {starting_dim},\n'
                f' possible dims:{set(self.allowed_dims.keys())}')


        self.first_dense_user = self.create_dense_entity_layers(self.first_modality_features.size(1), dropout_value)
        self.first_dense_item = self.create_dense_entity_layers(self.first_modality_features.size(1), dropout_value)

        self.second_dense_user = self.create_dense_entity_layers(self.second_modality_features.size(1), dropout_value)
        self.second_dense_item = self.create_dense_entity_layers(self.second_modality_features.size(1), dropout_value)

        self.third_dense_user = self.create_dense_entity_layers(self.third_modality_features.size(1), dropout_value)
        self.third_dense_item = self.create_dense_entity_layers(self.third_modality_features.size(1), dropout_value)


class GETAllNetworkTriConcat(ThreeSourceGETAllNetwork):

    def __init__(self, modality_features, dropout_value):
        super().__init__(modality_features, dropout_value)

        self.dense_user = torch.nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.dense_item = torch.nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # dense layers after concatenation and score computing
        self.compute_score = torch.nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
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

        first_users_features = self.first_modality_features[user_idx]
        first_items_features = self.first_modality_features[item_idx]

        second_users_features = self.second_modality_features[user_idx]
        second_items_features = self.second_modality_features[item_idx]

        third_users_features = self.third_modality_features[user_idx]
        third_items_features = self.third_modality_features[item_idx]

        first_users_features = self.first_dense_user(first_users_features)
        first_items_features = self.first_dense_item(first_items_features)

        second_users_features = self.second_dense_user(second_users_features)
        second_items_features = self.second_dense_item(second_items_features)

        third_users_features = self.third_dense_user(third_users_features)
        third_items_features = self.third_dense_item(third_items_features)

        # concat users and items and reduce dimensionalities
        user = self.dense_user(torch.cat([first_users_features, second_users_features, third_users_features], dim=-1))
        item = self.dense_item(torch.cat([first_items_features, second_items_features, third_items_features], dim=-1))

        # compute the recommendation scores
        out = self.compute_score(torch.cat([user, item], dim=-1))
        
        return out


# self attention between user and item for each source (graph, text, image)
# than cross attention to fuse the resulting 3 embeddings
class GETAllNetworkTriSelfSourcesCross(ThreeSourceGETAllNetwork):

    def __init__(self, modality_features, dropout_value):
        super().__init__(modality_features, dropout_value)

        # query, key, value for graph embeddings
        self.graph_qkv_1 = nn.Linear(256, 3*256)

        # query, key, value for text embeddings
        self.text_qkv_1 = nn.Linear(256, 3*256)

        # query, key, value for image embeddings
        self.img_qkv_1 = nn.Linear(256, 3*256)

        # linear layers for weighted_graph
        self.linear_weighted_graph = torch.nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # linear layers for weighted_text
        self.linear_weighted_text = torch.nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # linear layers for weighted_img
        self.linear_weighted_img = torch.nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.linear_key_value = torch.nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # cross attention between graph, text and img embeddings
        self.final_q = nn.Linear(64, 64)
        self.final_kv = nn.Linear(64, 2*64)

        # dense layers after concatenation and score computing
        self.compute_score = torch.nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
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
    def compute_attention(query, key, value):

        dim = query.size()[-1]
        attention_weights = nn.functional.softmax(torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim), dim=-1)
        weighted_embedding = torch.matmul(attention_weights, value)
        return weighted_embedding
    
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

        third_users_features = self.third_modality_features[user_idx]
        third_items_features = self.third_modality_features[item_idx]

        first_users_features = self.first_dense_user(first_users_features)
        first_items_features = self.first_dense_item(first_items_features)

        second_users_features = self.second_dense_user(second_users_features)
        second_items_features = self.second_dense_item(second_items_features)

        third_users_features = self.third_dense_user(third_users_features)
        third_items_features = self.third_dense_item(third_items_features)

        # for each source, self attention between entities

        # first self attention on graph embeddings
        query, key, value = self.graph_qkv_1(torch.stack([first_users_features, first_items_features], dim=1)).chunk(3, dim=-1)
        weighted_graph = self.compute_attention(query, key, value).sum(dim=1)

        # second self attention on text embeddings
        query, key, value = self.text_qkv_1(torch.stack([second_users_features, second_items_features], dim=1)).chunk(3, dim=-1)
        weighted_text = self.compute_attention(query, key, value).sum(dim=1)

        # third self attention on img embeddings
        query, key, value = self.img_qkv_1(torch.stack([third_users_features, third_items_features], dim=1)).chunk(3, dim=-1)
        weighted_img = self.compute_attention(query, key, value).sum(dim=1)

        # linear layers to reduce dimensionality
        weighted_graph = self.linear_weighted_graph(weighted_graph)
        weighted_text = self.linear_weighted_text(weighted_text)
        weighted_img = self.linear_weighted_img(weighted_img)

        # concat and linear layers for key and value based on text and img
        linear_key_value = self.linear_key_value(torch.concat([weighted_text, weighted_img], dim=-1))

        # cross attention with graph as query, and projected concat as key value
        query = self.final_q(weighted_graph)
        key, value = self.final_kv(linear_key_value).chunk(2, dim=-1)
        weighted_final = self.compute_attention(query, key, value)

        # no need to concat the resulting embeddings

        # compute the recommendation scores
        out = self.compute_score(weighted_final)

        return out
