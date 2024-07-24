import clayrs.content_analyzer as ca
import clayrs.evaluation as eva
import pandas as pd
import os
import warnings
from tqdm import tqdm


warnings.filterwarnings("ignore")

def eval(dataset='', modality='', ks=[5], relevant_threshold=1):


    # files to be evaluated
    preds_fold = f'../all_preds/rq3_sensitivity/{dataset}/{modality}'
    prediction_list = [x for x in os.listdir(f'{preds_fold}/') if '.DS_Store' not in x]
    print('Predictions to be evaluated:', prediction_list)

    # train data
    train_data = ca.CSVFile(os.path.join('datasets', f"{dataset}", "train.tsv"), separator="\t")
    train_ratings = ca.Ratings(train_data)

    # test data
    test_data = ca.CSVFile(os.path.join('datasets', f"{dataset}", "test.tsv"), separator="\t")
    test_ratings = ca.Ratings(test_data)
    
    dict_results = {}

    for _, prediction in enumerate(prediction_list):
        metric_list = []

        # eval at each k
        for k in ks:
            metric_list.extend([
                eva.PrecisionAtK(k=k, relevant_threshold=relevant_threshold),
                eva.RecallAtK(k=k, relevant_threshold=relevant_threshold),
                eva.FMeasureAtK(k=k, relevant_threshold=relevant_threshold),
                eva.NDCGAtK(k=k),
                eva.GiniIndex(k=k),
                eva.EPC(k=k, original_ratings=train_ratings, ground_truth=test_ratings),
                eva.APLT(k=k, original_ratings=train_ratings),
                
            ])

        eval_summary = ca.CSVFile(os.path.join(f'{preds_fold}', f"{prediction}"), separator="\t")

        truth_list = [test_ratings]
        rank_list = [ca.Rank(eval_summary)]

        em = eva.EvalModel(
            pred_list=rank_list,
            truth_list=truth_list,
            metric_list=metric_list
        )
        
        # compute metrics and save user results for statistical tests
        sys_result, users_result = em.fit()
        sys_result = sys_result.loc[['sys - mean']]
        sys_result.reset_index(drop=True, inplace=True)
        sys_result['model'] = prediction
        sys_result.columns = [x.replace(" - macro", "") for x in sys_result.columns]
        cols = list(sys_result.columns)
        cols = cols[-1:] + cols[:-1]
        sys_result = sys_result.loc[:, cols]
        dict_results[prediction] = sys_result


    results = pd.concat([v for v in dict_results.values()]).reset_index(drop=True).sort_values(by=['model'], ascending=[True])
    print(results)

    results.to_csv(f'results/rq3/{dataset}_{modality}_results.tsv', index=False, sep='\t')
    # results.to_csv(f'{prediction_dest_results}/{dataset}_{prediction_type_model}_results.tsv', index=False, sep='\t')

    


relevant_threshold=1

# dataset ranges in ['ml1m', 'dbbook']
# sentivity is related to all the possible parameters in the folder for that dataset 
# (it may vary based on the available modalities for that dataset)

print('in exec')

for dataset in ['ml1m', 'dbbook']:
    modalities = [x for x in os.listdir(f'rq3_sensitivity/{dataset}') if '.DS_Store' not in x]
    for modality in modalities:
        print(dataset, modality)
        eval(dataset, modality, ks=[5], relevant_threshold=1)

