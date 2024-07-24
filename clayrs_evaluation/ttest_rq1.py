import clayrs.content_analyzer as ca
import clayrs.evaluation as eva
import pandas as pd
import os
import warnings
from tqdm import tqdm
import clayrs.evaluation.statistical_test as st

from scipy import stats
import math


warnings.filterwarnings("ignore")

def stat_eval(dataset='', ks=[5], relevant_threshold=1):

    # if the eval has been already performed, skip it
    if os.path.exists(f'stat_{dataset}_results.tsv') and False:
        print(f'Skipping stat {dataset}')
        return

    # files to be evaluated
    preds_fold = f'../all_preds/rq1_baselines/{dataset}/stat'
    prediction_list = [x for x in os.listdir(f'{preds_fold}/') if '.DS_Store' not in x]
    print('Predictions to be evaluated:', prediction_list)

    # train data
    train_data = ca.CSVFile(os.path.join('datasets', f"{dataset}", "train.tsv"), separator="\t")
    train_ratings = ca.Ratings(train_data)

    # test data
    test_data = ca.CSVFile(os.path.join('datasets', f"{dataset}", "test.tsv"), separator="\t")
    test_ratings = ca.Ratings(test_data)
    
    dict_results = {}

    # list of user results dfs for ttest
    users_results = []


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
        users_results.append(users_result)


    # compute ttests
    p_values = dict()

    for i, pred_1 in enumerate(prediction_list):
        for j, pred_2 in enumerate(prediction_list):

            pred_1 = pred_1.replace('.tsv', '')
            pred_2 = pred_2.replace('.tsv', '')

            if not i == j and f'{pred_2}-{pred_1}' not in p_values:

                p_values[f'{pred_1}-{pred_2}'] = dict()

                for metric in users_results[i].columns:

                    res_1 = []
                    res_2 = []

                    for x, y in zip(list(users_results[i][metric]), list(users_results[j][metric])):
                        if not (math.isnan(x) or math.isnan(y)):
                            res_1.append(x)
                            res_2.append(y)

                    _, p_value = stats.ttest_rel(res_1, res_2)
                    p_values[f'{pred_1}-{pred_2}'][metric] = p_value

    ttest = pd.DataFrame.from_dict(p_values, orient='index')
    ttest.reset_index(inplace=True)
    ttest.rename(columns={'index': 'Predictions'}, inplace=True)

    ttest.to_csv(f'results/rq1/stat_rq1_{dataset}_results.tsv', index=False, sep='\t')

relevant_threshold=1

# dataset ranges in ['ml1m', 'dbbook']

for dataset in ['dbbook', 'ml1m']:
    stat_eval(dataset, ks=[5], relevant_threshold=1)

