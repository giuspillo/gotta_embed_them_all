import clayrs.content_analyzer as ca
import clayrs.evaluation as eva
import pandas as pd
import os
import warnings
from tqdm import tqdm


warnings.filterwarnings("ignore")

def eval(dataset='', prediction_type_model='', ks=[5], relevant_threshold=1):

    if prediction_type_model == 'baseline':
        preds_fold = f'../all_preds/rq1_baseline/{dataset}'
        prediction_dest_results = 'rq1'
    else:
        preds_fold = f'../all_preds/rq2_getall/{dataset}'
        prediction_dest_results = 'rq2'

    # files to be evaluated
    preds_fold = f'{prediction_type_model}/{dataset}'
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

    results.to_csv(f'{prediction_dest_results}/{dataset}_{prediction_type_model}_results.tsv', index=False, sep='\t')

    


relevant_threshold=1

# dataset ranges in ['ml1m', 'dbbook']
# prediction_type_model ranges in ['getall', 'baselines']

for dataset in ['dbbook', 'ml1m']:
    for prediction_type_model in ['getall', 'baselines']:

        eval(dataset, prediction_type_model, ks=[5], relevant_threshold=1)

