from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from os import path
import torch

# organized as list so that it is easy to automatically iterate 
# if you want to add other datasets, models, or embedding dimensions

np.random.seed(42)

# set dataset name, model of the GCN, dimension, num_layer, epochs, and only user-item flag
datasets = ['movielens']
emb_models = ['CompGCN']
emb_dims = [128, 192, 256, 384, 512]
n_layers = [1]
emb_epochs = 15
only_ui = False

log_file = open('log.txt', 'w')

for emb_model in emb_models:
    for emb_dim in emb_dims:
        for dataset in datasets:
            for n_layer in n_layers:

                # user-item + KG setting
                if not only_ui:

                    printline = dataset+' - '+emb_model+' - k='+str(emb_dim)+' - layers='+str(n_layer)
                    print('Starting ' + printline)
                    log_file.write('Starting' + printline)

                    folder = 'results/' + dataset+'_'+emb_model+'_k='+str(emb_dim) + 'n_layers='+str(n_layer)

                    train_path = dataset + '/pykeen_train.tsv'
                    test_path = dataset + '/pykeen_test.tsv'

                    checkpoint_name_file = "checkpoint_"+dataset+'_'+emb_model+'_k='+str(emb_dim)+'n_layers='+str(n_layer)

                # user-item only setting
                else:

                    printline = dataset+'_ui - '+emb_model+' - k='+str(emb_dim)+' - layers='+str(n_layer)
                    print('Starting ' + printline)
                    log_file.write('Starting' + printline)
                    
                    folder = 'results/' + dataset+'_ui_'+emb_model+'_k='+str(emb_dim)+'n_layers='+str(n_layers)

                    train_path = dataset + '/pykeen_train_user-item.tsv'
                    test_path = dataset + '/pykeen_test_user-item.tsv'

                    checkpoint_name_file = "checkpoint_"+dataset+'_ui_'+emb_model+'_k='+str(emb_dim)+'n_layers='+str(n_layer)

                # if embs already exists, skip
                if os.path.isfile(folder+'/embeddings.tsv'):
                    print('Existing embedding in ' + folder)
                    continue

                # otherwise train
                try:      

                    print('Starting learning:' + folder)
                    print("Starting learning:", printline)
                    

                    emb_training = TriplesFactory.from_path(
                        train_path,
                        create_inverse_triples=True,
                    )

                    emb_testing = TriplesFactory.from_path(
                        test_path,
                        entity_to_id=emb_training.entity_to_id,
                        relation_to_id=emb_training.relation_to_id,
                        create_inverse_triples=True,
                    )

                    result = pipeline(
                        training=emb_training,
                        testing=emb_testing,
                        model=emb_model,
                        model_kwargs=dict(embedding_dim=emb_dim,
                                          encoder_kwargs=dict(num_layers=n_layer)),
                        random_seed=42,
                        evaluation_fallback = True,
                        training_kwargs=dict(
                            num_epochs=emb_epochs,
                            checkpoint_name=checkpoint_name_file,
                            checkpoint_directory='checkpoints',
                            checkpoint_frequency=1
                        ),
                    )

                    if not os.path.exists(folder):
                        os.mkdir(folder)


                    torch.save(result, folder+'/pipeline_result.dat')

                    map_ent = pd.DataFrame(data=list(emb_training.entity_to_id.items()))
                    map_ent.to_csv(folder+'/entities_to_id.tsv', sep='\t', header=False, index=False)
                    map_ent = pd.DataFrame(data=list(emb_training.relation_to_id.items()))
                    map_ent.to_csv(folder+'/relations_to_id.tsv', sep='\t', header=False, index=False)


                    # save mappings
                    result.save_to_directory(folder, save_training=True, save_metadata=True)

                    # extract embeddings with gpu
                    entity_embedding_tensor = result.model.entity_representations[0](indices = None)
                    # save entity embeddings to a .tsv file (gpu)
                    df = pd.DataFrame(data=entity_embedding_tensor.cpu().data.numpy())

                    # extract embeddings with cpu
                    #entity_embedding_tensor = result.model.entity_representations[0](indices=None).detach().numpy()
                    # save entity embeddings to a .tsv file (cpu)
                    #df = pd.DataFrame(data=entity_embedding_tensor.astype(float))

                    outfile = folder + '/embeddings.tsv'
                    df.to_csv(outfile, sep='\t', header=False, index=False)

                    print('Completed ' + printline)
                    log_file.write('Completed\n')
                
                except Exception as e:

                    print('An error occoured in ' + printline)
                    log_file.write('An error occoured in ' + printline + '\n')
                    print(e)
                    log_file.write(str(e)+'\n')

log_file.flush()
log_file.close()
