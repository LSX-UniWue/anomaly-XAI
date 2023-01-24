import os
import warnings
import shap
import numpy as np
import pandas as pd


def shap_explain(predict_fn, data, baseline=None, out_file_path=None, **shap_kwargs):
    """
    SHAP for the tabular regression task

    SHAP behavior on Regression:
    shap_values[0].sum(1)      + explainer.expected_value - model.predict(X, raw_score=True) ~ 0
    attribution sum per sample + expected value           - output                           ~ 0
    """
    # Background dataset for "default" values when "removing"/perturbing features:
    if baseline is None:  # with no baseline: use 0 for missing features
        baseline = np.zeros(data.shape[1]).reshape(1, data.shape[1])
    elif len(baseline.shape) > 1 and baseline.shape[0] > 100:
        baseline = shap.kmeans(baseline, k=20)
    elif len(baseline.shape) < 2:
        baseline = baseline.reshape(1, *baseline.shape)

    shap_xai = shap.KernelExplainer(predict_fn, baseline, **shap_kwargs)

    # get shap values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = shap_xai.shap_values(data)

    print(f"SHAP expected_value: {shap_xai.expected_value}")

    if out_file_path:
        if isinstance(data, pd.Series):
            out_df = pd.DataFrame(shap_values, columns=[data.name], index=data.index.values).T
            out_df['expected_value'] = shap_xai.expected_value
        else:
            out_df = pd.DataFrame(shap_values, index=data.index, columns=data.columns)
        if os.path.exists(out_file_path):
            old_df = pd.read_csv(out_file_path, header=0, index_col=0)
            out_df.columns = out_df.columns.astype(str)  # read_csv reads column names as str by default
            out_df = pd.concat([old_df, out_df])
        out_df.to_csv(out_file_path)

    return shap_xai, shap_values


def explain_anomalies(X_anomalous, predict_fn, X_benign, background, out_file_path=None, model_to_optimize=None):
    """
    Generates Shap explanations with different background datasets depending on background, saves to out_template
    :param X_anomalous:        pd.DataFrame including data to explain
    :param predict_fn:         forward function to explain
    :param X_benign:           pd.DataFrame including benign data to sample background data from
    :param background:         Option for background generation: May be one of:
                               'mean':                 Takes mean of X_train data as background
                               'zeros':                Takes zero vector as background
                               'NN':                   Calculates k nearest neighbors in X_benign with euclidean distance
                               'optimized'             optimized input in vicinity of the anomaly
    :param out_file_path:      Str path to output .csv file
    :param model_to_optimize:  pytorch model to optimize input for when choosing an optimized background
    """
    print("Calculating SHAP values...", flush=True)
    if background in ['zeros', 'mean', 'NN']:
        # these have one background point for each sample to explain
        for idx, (anom_id, row) in enumerate(X_anomalous.iterrows()):
            shap_explain(predict_fn=predict_fn,
                         data=row,
                         baseline=X_benign[idx],
                         out_file_path=out_file_path)

    elif background in ['optimized']:
        # These need multiple optimizations for each sample that need to be computed in SHAP
        for idx, (anom_id, row) in enumerate(X_anomalous.iterrows()):
            shap_explain(predict_fn=predict_fn,
                         data=row,
                         baseline=pd.DataFrame(np.full(shape=(1, X_benign.shape[1]), fill_value=np.inf), columns=X_benign.columns),
                         out_file_path=out_file_path,
                         full_model=model_to_optimize,
                         dynamic_background=background)

    else:
        raise ValueError("Variable background not one of ['mean', 'zeros', 'NN', 'optimized']")
