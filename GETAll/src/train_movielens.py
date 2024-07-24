from __future__ import annotations

import os
import torch
from torchsummary import summary

import numpy as np
import pandas as pd
import torch.nn.functional as fun
import pickle as pkl
import random

from tqdm import tqdm
from typing import Type
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from termcolor import colored
import sys
import time

from getall_dataset import GETAllDataset
from getall_basic import GETAllNetwork, GETAllNetworkBasic
from getall_double_sources import GETAllNetworkConcat, GETAllNetworkMerge
from getall_tri_sources import GETAllNetworkTriConcat, GETAllNetworkTriSelfSourcesCross
from getall_four_sources import GETAllNetworkFourConcat, GETAllNetworkFourSelfSourcesCross
from getall_five_sources import GETAllNetworkFiveConcat, GETAllNetworkFiveSelfSourcesCross


THIS_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def train(train_ratings: pd.DataFrame, batch_size: int, epochs: int, device: str, model_class: Type[GETAllNetwork],
           modality_features: np.array, weights: list, dropout_value: float):

    train_ds = GETAllDataset(train_ratings)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    # init the corresponding GETALL network using item and user features
    model = model_class(
        modality_features = modality_features,
        dropout_value = dropout_value
    ).to(device)

    if weights is not None:
        model = model.import_weights(weights)

    optimizer = Adam(model.parameters())
    # optimizer = SGD(model.parameters(), lr=0.001)

    print(colored('Training set with model '+str(type(model))+' and '+str(len(modality_features))+' features', 'green'))

    for epoch in range(epochs):

        train_loss = 0
        tqdm_bar = tqdm(train_dl)

        for i, batch in enumerate(tqdm_bar):

            optimizer.zero_grad()

            user_idx = batch[0].int()
            item_idx = batch[1].int()

            # model input is a tuple (user, item) idxs where the idx refers to the embedding stored
            # inside the first parameters of the network (item_features and user_features)
            model_input = (
                user_idx,
                item_idx,
                dropout_value
            )

            score = model(model_input)
            loss = fun.binary_cross_entropy(score.flatten().to(device), batch[2].to(device).float())

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            tqdm_bar.set_description(
                f"[Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss / (i + 1)}]"
            )

    model.cpu().eval()
    print('Finished')

    return model


@torch.no_grad()
def test(model: GETAllNetwork, test_ratings: pd.DataFrame, batch_size: int, device: str, dropout_value: float):

    model.to(device)

    test_ds = GETALLDataset(test_ratings)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size)
    tqdm_bar = tqdm(test_dl)

    user_idxs = []
    item_idxs = []
    scores = []

    for i, batch in enumerate(tqdm_bar):

        user_idx = batch[0].int()
        item_idx = batch[1].int()

        model_input = (
            user_idx,
            item_idx,
            dropout_value
        )

        score = model(model_input)

        user_idxs.append(user_idx)
        item_idxs.append(item_idx)
        scores.append(score.squeeze())

    user_idxs = torch.hstack(user_idxs).int().numpy()
    item_idxs = torch.hstack(item_idxs).int().numpy()
    scores = torch.hstack(scores).float().cpu().numpy()

    # create the DataFrame where each row is (user, item, score)
    predictions = pd.DataFrame.from_records(
        zip(user_idxs, item_idxs, scores),
        columns=['user', 'item', 'score'],
        index=None
    )

    predictions = predictions.sort_values(by=['user', 'score'], ascending=[True, False])
    
    return predictions


if __name__ == "__main__":

    getall_network_class = GETAllNetworkBasic
    
    train_ratings = pd.read_table(os.path.join(RAW_DATA_DIR, 'movielens', 'train.tsv'),
                                  names = ['user', 'item', 'score'], sep='\t')
    print(f'New train len {len(train_ratings)}')

    test_ratings = pd.read_table(os.path.join(RAW_DATA_DIR, 'movielens', 'test.tsv'),
                                  names = ['user', 'item', 'score'], sep='\t')
    print(f'New test len {len(test_ratings)}')

    # get unique users and map them to indices (0, 1, 2, ...)
    unique_users = np.unique(train_ratings['user'])
    unique_users_indices = np.arange(0, unique_users.shape[0])

    # get unique items and map them to indices (0, 1, 2, ...)
    unique_items = np.unique(train_ratings['item'])
    unique_items_indices = np.arange(0, unique_items.shape[0])

    emb_folders = ['data/embeddings/movielens/graph/',
                'data/embeddings/movielens/text/',
                'data/embeddings/movielens/images/',
                'data/embeddings/movielens/video/',
                'data/embeddings/movielens/audio/',
                ]


    tensors_dict = dict()
    total_size = 0
    for folder in emb_folders:
        embs = [os.path.join(folder, x) for x in os.listdir(folder)]
        for emb in embs:
            if '.pkl' in emb:
                with open(emb, 'rb') as f:
                    emb_features = pkl.load(f)
                    emb_features = dict(sorted(emb_features.items()))
                    emb_features_tensor = torch.from_numpy(np.stack(list(emb_features.values())))
                    total_size += sys.getsizeof(emb_features_tensor)
                    name_model = emb.split('/')[-1].replace('.pkl','')
                    tensors_dict[name_model] = emb_features_tensor

    print(colored('Embeddings loaded. Total size: ' + str(total_size), 'blue'))

    # get any source and produce the mapping - we choose the graph, but it is the same
    # due to the fact that the embeddings are always for the same entities (9117)
    # and we are sorting the keys
    source_features = dict(sorted((pkl.load(open('data/embeddings/movielens/graph/compgcn_384_n1.pkl', 'rb'))).items()))
    source_mapping_indeces = np.arange(0, len(source_features))
    source_mapping = dict(zip(source_features.keys(), source_mapping_indeces))

    # apply the mapping to the train and test ratings
    train_ratings['user'] = train_ratings['user'].map(source_mapping)
    train_ratings['item'] = train_ratings['item'].map(source_mapping)
    test_ratings['user'] = test_ratings['user'].map(source_mapping)
    test_ratings['item'] = test_ratings['item'].map(source_mapping)


    feature_list = [ ['compgcn_384_n1', 'all-MiniLM-L12-v2_dictionary', 'vit_cls', 'r2p1d', 'vggish'] ]
    getall_models = [GETAllNetworkFiveSelfSourcesCross]    
    dropout_values = [ 0.5 ]
    setting_names = []
    setting_sources = ['uni', 'double', 'tri', 'four', 'five']
    
    for i, x in enumerate(feature_list):
        
        setting_names.append(f'drop{str(dropout_values[i])}_' + f'{setting_sources[len(x) - 1]}_' + f'{getall_models[i].__name__}_' + '_'.join(x))

    # check lengths
    check_lengths = len(feature_list) == len(dropout_values) and \
        len(dropout_values) == len(getall_models) and \
        len(getall_models) == len(setting_names)

    if not check_lengths:
        print(colored('Lengths not matching', 'red'))
        exit(0)

    print(f'Len feature list: {len(feature_list)}')
    print(f'Len dropout values: {len(dropout_values)}')
    print(f'Len getall models: {len(getall_models)}')
    print(f'Len setting names: {len(setting_names)}')


    for i, config in enumerate(feature_list):

            print(i, config)

            # set seed at each iteration
            set_seed(42)

            features = [tensors_dict[model] for model in config]
            getall_model = getall_models[i]
            dropout_value = dropout_values[i]
            setting_name = setting_names[i]

            print(f'Len features: {len(features)}')
            print(f'Setting name: {setting_name}')
            print(f'GETALL model:  {getall_model}')
            print(f'Dropout value: {dropout_value}')

            preds_folder = f'reports/predictions/{setting_sources[len(x) - 1]}/'
            model_folder = f'reports/models/{setting_sources[len(x) - 1]}/'

            os.makedirs(preds_folder, exist_ok=True)
            os.makedirs(model_folder, exist_ok=True)

            preds_file = preds_folder + setting_name + '_top5.tsv'
            model_name = model_folder + setting_name + '.pth'

            # print current setting
            print(colored('Setting: ' + setting_name, 'green'))

            # get device
            device = 'cpu'
            if torch.cuda.is_available():
                device = 'cuda:0'

            print(colored('Using ' + device, 'blue'))

            # set some hyperparameters
            epochs = 30
            train_batch_size = 512
            test_batch_size = 512
            dropout_value = dropout_values[i]

            force = True

            if os.path.exists(model_name) and force is False:

                # check if the predictions already exists
                if os.path.exists(preds_file):
                    print(colored('The predictions already exists, skipping.','green'))
                    continue

                # load the model if existing
                print(colored('Model \'' + model_name + '\' found, it will be loaded.', 'green'))
                trained_model = torch.load(model_name)

            else:

                # print the setting
                print(colored('Model \'' + model_name + '\' not found, it will be trained.', 'green'))

                # train the model
                trained_model = train(train_ratings, train_batch_size, epochs, device, 
                    getall_model, features, weights=None, dropout_value = dropout_value)
                
                # save the model
                torch.save(trained_model, model_name)
                print(colored('Model saved','green'))

            # start testing the model
            print(colored('Testing the model:', 'blue'))
            predictions = test(trained_model, test_ratings, test_batch_size, device, dropout_value = dropout_value)

            # remap user and items to original ids
            list_mapping = list(source_features)
            predictions['user'] = predictions['user'].map(lambda x: list_mapping[x])
            predictions['item'] = predictions['item'].map(lambda x: list_mapping[x])

            # Group the dataframe by user_id so that applying the .head function will return the first results
            # for each user
            grouped_predictions = predictions.groupby('user', as_index=False)

            top5 = grouped_predictions.head(5)
            # top10 = grouped_predictions.head(10)
            
            top5.to_csv(preds_file, sep='\t', index=False, header=None)
            # top10.to_csv('predictions/'+setting_name+'_top10.pth', sep='\t', index=False)
            
            print(colored('Finished!', 'blue'))
