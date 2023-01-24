import numpy as np
import pandas as pd
from lime import lime_tabular


def explain_anomalies(X_anomalous,
                      X_benign,
                      xai_type,
                      detector,
                      out_template,
                      **kwargs):
    # Create and train the LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(training_data=X_benign.values,  # need numpy array, not pandas df
                                                  feature_names=X_benign.columns.values.tolist(),
                                                  class_names='Fraud Score',
                                                  mode='regression',
                                                  categorical_features=list(range(len(X_benign.columns))),  # all
                                                  random_state=42)  # reproducibility

    lime_explanation = np.empty(X_anomalous.shape)
    for i in range(X_anomalous.shape[0]):
        # Explain all instances in the X_anomalous data
        exp = explainer.explain_instance(data_row=X_anomalous.iloc[i].to_numpy(),
                                         predict_fn=detector.score_samples,
                                         num_features=X_anomalous.shape[1])
        lime_exp = list(exp.local_exp.items())[0]
        lime_sample = lime_exp[1]  # LIME feature scores
        lime_sample = sorted(lime_sample, key=lambda x: x[0])  # Sort
        lime_sample = [x[1] for x in lime_sample]
        lime_explanation[i] = lime_sample

    if out_template:
        pd.DataFrame(lime_explanation,
                     columns=X_anomalous.columns,
                     index=X_anomalous.index).to_csv(out_template.format(xai_type))

    return lime_explanation
