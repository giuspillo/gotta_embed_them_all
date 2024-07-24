from __future__ import annotations

import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import math

import pandas as pd


class GETAllDataset(data.Dataset):
    def __init__(self, train_ratings: pd.DataFrame):

        self.train_ratings_users = np.array(train_ratings['user'])
        self.train_ratings_items = np.array(train_ratings['item'])
        self.train_ratings_scores = np.array(train_ratings['score'])

    def __len__(self):
        return self.train_ratings_users.shape[0]

    def __getitem__(self, idx):

        user_idx = self.train_ratings_users[idx]
        item_idx = self.train_ratings_items[idx]
        rating = self.train_ratings_scores[idx]

        return user_idx, item_idx, rating