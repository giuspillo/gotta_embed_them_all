import pickle as pkl
import numpy as np

train = 'train.tsv'

with open(train, 'r') as fin:
    
    user_pref = dict()
    
    for line in fin:
        user, item, rating = [int(x) for x in line.split('\t')]
            
        if rating == 1:
            if user not in user_pref:
                user_pref[user] = set()
            user_pref[user].add(item)

emb_paths = ['embs/vit_csl_embs.pkl',] # add any other text model you want!

for emb_path in emb_paths:

    embs = pkl.load(open(emb_path, 'rb'))


    for user in user_pref:
        list_emb = list()
        for item in user_pref[user]:
            if item in embs:
                list_emb.append(embs[item])
        centroid = np.mean(list_emb, axis=0)
        embs[user] = centroid

    new_name = emb_path.split('.')[0]+'_user-item.pkl'

    pkl.dump(embs, open(new_name, 'wb'))
    print('Completed', emb_path)