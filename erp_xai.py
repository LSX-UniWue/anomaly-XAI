
import os
import functools
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from xai.util import xai_to_categorical, tabular_reference_points
from outputs.models.util import load_best_detector
from data.erpDataset import ERPDataset


def get_expl_scores(explanation, gold_standard, dataset, score_type='auc_roc'):
    """Calculate AUC-ROC score for each sample individually, report mean and std"""
    scores = []
    for i, row in explanation.iterrows():
        # Explanation values for each feature treated as likelihood of anomalous feature
        #  -aggregated to feature-scores over all feature assignments
        #  -flattened to match shape of y_true
        #  -inverted, so higher score means more anomalous
        y_score = xai_to_categorical(expl_df=pd.DataFrame(explanation.loc[i]).T,
                                     dataset=dataset).values.flatten() * -1
        # Calculate score
        if score_type == 'auc_roc':
            scores.append(roc_auc_score(y_true=gold_standard.loc[i], y_score=y_score))
        elif score_type == 'cosine_sim':
            scores.append(cosine_similarity(gold_standard.loc[i].values.reshape(1, -1), y_score.reshape(1, -1))[0, 0])
        else:
            raise ValueError(f"Unknown score_type '{score_type}'")

    return np.mean(scores), np.std(scores)


def evaluate_expls(background,
                   train_path,
                   test_path,
                   gold_standard_path,
                   expl_folder,
                   xai_type,
                   out_path,
                   data):
    """Calculate AUC-ROC score of highlighted important features"""
    run_name = Path(train_path).stem + "_" + Path(test_path).stem
    expl = pd.read_csv(Path(expl_folder) / '{}_{}.csv'.format(xai_type, background + '_' + run_name),
                       header=0, index_col=0)
    if 'expected_value' in expl.columns:
        expl = expl.drop('expected_value', axis=1)
    # Load gold standard explanations and convert to pd.Series containing
    # anomaly index & list of suspicious col names as values
    gold_expl = pd.read_csv(gold_standard_path, header=0, index_col=0, encoding='UTF8')
    gold_expl = (gold_expl == 'X').iloc[:, :-5]
    to_check = data.get_frauds().index.tolist()

    assert len(to_check) == gold_expl.shape[0], \
        f"Not all anomalies found in explanation: Expected {gold_expl.shape[0]} but got {len(to_check)}"

    # # what if expl is random uniform noise?
    # noise = pd.DataFrame(np.random.uniform(low=0, high=1, size=expl.loc[to_check].shape),
    #                      columns=expl.columns,
    #                      index=expl.loc[to_check].index)
    # # what if expl is random uniform noise, multiplied by input?
    # noise = noise * data.preprocessed_data['X_test'].iloc[to_check]

    # watch out for naming inconsistency! The dataset=data that get_expl_scores gets is an ERPDataset instance!
    roc_mean, roc_std = get_expl_scores(explanation=expl.loc[to_check],
                                        gold_standard=gold_expl.loc[to_check],
                                        score_type='auc_roc',
                                        dataset=data)
    cos_mean, cos_std = get_expl_scores(explanation=expl.loc[to_check],
                                        gold_standard=gold_expl.loc[to_check],
                                        score_type='cosine_sim',
                                        dataset=data)

    out_dict = {'xai': xai_type,
                'variant': background,
                f'ROC': roc_mean,
                f'ROC-std': roc_std,
                f'Cos': cos_mean,
                f'Cos-std': cos_std}
    [print(key + ':', val) for key, val in out_dict.items()]

    # save outputs to combined result csv file
    if out_path:
        if os.path.exists(out_path):
            out_df = pd.read_csv(out_path, header=0)
        else:
            out_df = pd.DataFrame()
        out_df = out_df.append(out_dict, ignore_index=True)
        out_df.to_csv(out_path, index=False)
    return out_dict


def explain_anomalies(train_path,
                      test_path,
                      compare_with_gold_standard,
                      expl_folder,
                      xai_type='shap',
                      numeric_preprocessing='bucket',
                      background='zeros',
                      out_path=None,
                      **kwargs):
    """
    :param train_path:      Str path to train dataset
    :param test_path:       Str path to test dataset
    :param compare_with_gold_standard:  Whether to evaluate the explanations vs the gold standard
    :param expl_folder:     Str path to folder to write/read explanations to/from
    :param numeric_preprocessing:   Str type of numeric preprocessing, one of ['buckets', 'log10', 'zscore', 'None']
    :param background:      Option for background generation: May be one of:
                            'zeros':                Zero vector as background
                            'mean':                 Takes mean of X_train data through k-means (analog to SHAP)
                            'NN':                   Finds nearest neighbor in X_train
                            'optimized':            Optimizes samples while keeping one input fixed
    :param kwargs:          Additional keyword args directly for numeric preprocessors during data loading
    """
    run_name = Path(train_path).stem + "_" + Path(test_path).stem

    data = ERPDataset(train_path=train_path,
                      test_path=test_path,
                      numeric_preprocessing=numeric_preprocessing,
                      categorical_preprocessing='ordinal' if 'ordinal' in xai_type else 'onehot',
                      keep_index=True,
                      **kwargs)

    X_train, _, _, _, X_test, y_test, _, _ = data.preprocessed_data.values()

    # find gold standard explanations for anomalous cases
    ds_file = Path(test_path).stem + "_expls.csv"
    gold_expl_path = f'data/erp_fraud/{ds_file}'
    X_expl = data.get_frauds().sort_index()

    print('Loading detector...')
    detector = load_best_detector(model='AE')

    # Generating explanations
    if not os.path.exists(os.path.join(expl_folder,
                                       '{}_{}.csv'.format(xai_type, background + '_' + run_name))):
        print("Generating explanations...")
        out_template = os.path.join(expl_folder, '{{}}_{}.csv'.format(background + '_' + run_name))

        if xai_type in ['shap']:
            import xai.xai_shap

            predict_fn = detector.score_samples
            if background in ['zeros', 'mean', 'NN']:
                detector.to('cpu')
                reference_points = tabular_reference_points(background=background,
                                                            X_expl=X_expl.values,
                                                            X_train=X_train.values,
                                                            predict_fn=functools.partial(detector.score_samples,
                                                                                         output_to_numpy=False))
            else:
                reference_points = X_train

            xai.xai_shap.explain_anomalies(X_anomalous=X_expl,
                                           predict_fn=predict_fn,
                                           X_benign=reference_points,
                                           background=background,
                                           model_to_optimize=detector,
                                           out_file_path=out_template.format(xai_type))

        elif xai_type in ['lime']:
            import xai.xai_lime
            xai.xai_lime.explain_anomalies(X_anomalous=X_expl,
                                           X_benign=X_train,
                                           xai_type=xai_type,
                                           detector=detector,
                                           out_template=out_template)

        elif xai_type in ['captum_deeplift', 'captum_intgrad', 'captum_lrp', 'captum_gradient', 'captum_grad_input']:
            import xai.xai_captum

            if xai_type in ['captum_intgrad']:  # approach needs a single background point per sample to be explained
                reference_points = tabular_reference_points(background=background,
                                                            X_expl=X_expl.values,
                                                            X_train=X_train.values,
                                                            predict_fn=functools.partial(detector.score_samples,
                                                                                         output_to_numpy=False))
            else:
                reference_points = None

            def predict_fn(X, detector):
                y = detector.score_samples(X, output_to_numpy=False)
                return y

            xai.xai_captum.explain_anomalies(X_anomalous=X_expl,
                                             reference_points=reference_points,
                                             xai_type=xai_type,
                                             model=detector,
                                             predict_fn=functools.partial(predict_fn, detector=detector),
                                             out_template=out_template,
                                             target=None,
                                             device='cpu')

        else:
            raise ValueError(f'Unknown xai_type: {xai_type}')

    if compare_with_gold_standard:
        print('Evaluating explanations...')
        out_dict = evaluate_expls(background=background,
                                  train_path=train_path,
                                  test_path=test_path,
                                  gold_standard_path=gold_expl_path,
                                  expl_folder=expl_folder,
                                  xai_type=xai_type,
                                  out_path=out_path,
                                  data=data)  # ERPDataset class instance
        return out_dict


if __name__ == '__main__':
    """
    Argparser needs to accept all possible param_search arguments, but only passes given args to params.
    """

    parser = ArgumentParser()
    parser.add_argument(f'--shard_data', type=int, default=None)
    args_dict = vars(parser.parse_args())

    # ['captum_deeplift', 'captum_intgrad', 'captum_lrp', 'captum_gradient', 'captum_grad_input', 'shap', 'lime']
    xai_type = 'captum_intgrad'

    # ['zeros', 'mean', 'NN', 'optimized']
    backgrounds = ['mean']

    train_path = './data/erp_fraud/normal_2.csv'
    test_path = './data/erp_fraud/fraud_3.csv'

    compare_with_gold_standard = True
    add_to_summary = True

    expl_folder = './outputs/explanation/erp_dataset'
    out_path = './outputs/explanation/erp_summary.csv' if add_to_summary else None

    for background in backgrounds:
        explain_anomalies(train_path=train_path,
                          test_path=test_path,
                          compare_with_gold_standard=compare_with_gold_standard,
                          expl_folder=expl_folder,
                          xai_type=xai_type,
                          numeric_preprocessing='buckets',
                          background=background,
                          out_path=out_path)
