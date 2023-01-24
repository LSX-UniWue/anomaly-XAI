import functools

import tqdm
import numpy as np
import pandas as pd
import sklearn

from data.preprocessing import (CategoricalOneHotPreprocessor,
                                CategoricalOrdinalPreprocessor,
                                NumericalQuantizationPreprocessor)


def scores_to_categorical(data, categories):
    """np.concatenate(data_cat, data[:, 29:])
    Slims a data array by adding column values of rows together for all column pairs in list categories.
    Used for summing up scores that were calculated for one-hot representation of categorical features.
    Gives a score for each categorical feature.
    :param data:        np.array of shape (samples, features) with scores from one-hot features
    :param categories:  list with number of features that were used for one-hot encoding each categorical feature
                        (as given by sklearn.OneHotEncoder.categories_)
    :return:
    """
    data_cat = np.zeros((data.shape[0], len(categories)))
    for i, cat in enumerate(categories):
        data_cat[:, i] = np.sum(data[:, cat], axis=1)
    if data.shape[1] > len(categories):  # get all data columns not in categories and append data_cat
        categories_flat = [item for sublist in categories for item in sublist]
        data_cat = np.concatenate((data[:, list(set(range(data.shape[1])) ^ set(categories_flat))], data_cat), axis=1)
    return data_cat


def create_mapping(dataset):
    counter = 0
    mapping_list = []

    num_prep = dataset.preprocessed_data['num_prep']
    cat_prep = dataset.preprocessed_data['cat_prep']

    if isinstance(cat_prep, CategoricalOneHotPreprocessor):
        for cat_mapping in cat_prep.encoder.category_mapping:
            mapping_list.append(list(range(counter, counter + cat_mapping['mapping'].size - 1)))
            counter += cat_mapping['mapping'].size - 1  # -1 because of double nan handling
    elif isinstance(cat_prep, CategoricalOrdinalPreprocessor):
        for _ in dataset.cat_cols:
            mapping_list.append([counter])
            counter += 1
    else:
        raise ValueError(f"Unknown categorical preprocessing: {type(cat_prep).__name__}")

    if isinstance(num_prep, NumericalQuantizationPreprocessor):
        for _ in range(num_prep.encoder.n_bins_.size):
            n_buckets = num_prep.encoder.n_bins + 1
            mapping_list.append(list(range(counter, counter + n_buckets)))
            counter += n_buckets
    else:
        for _ in dataset.num_cols:
            mapping_list.append([counter])
            counter += 1

    return mapping_list


def xai_to_categorical(expl_df, dataset=None):
    """
    Converts XAI scores to categorical values and adds column names
    """
    cat_cols = create_mapping(dataset)
    col_names = dataset.get_column_names()
    expls_joint = scores_to_categorical(expl_df.values, cat_cols)
    return pd.DataFrame(expls_joint, index=expl_df.index, columns=col_names)


def image_reference_points(background, X_expl, mvtec_data=None, predict_fn=None, device=None):
    if background == 'zeros':  # zero vector, default
        reference_points = np.zeros(X_expl.shape)

    elif background == 'NN':  # nearest neighbor in the normal training data
        if mvtec_data is None:
            raise ValueError("background 'NN' requires mvtec data object as input at variable 'mvtec_data'")
        from sklearn.neighbors import NearestNeighbors
        X_train, _, _ = mvtec_data.get_full_dataset('train')
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_train.reshape(X_train.shape[0], -1))
        neighbor_dist, neighbor_idx = nbrs.kneighbors(X=X_expl.reshape(X_expl.shape[0], -1), n_neighbors=1, return_distance=True)
        reference_points = X_train[neighbor_idx.flatten()]

    elif background == 'mean':  # mean training data point
        if mvtec_data is None:
            raise ValueError("background 'mean' requires mvtec data object as input at variable 'mvtec_data'")
        X_train, _, _ = mvtec_data.get_full_dataset('train')
        reference_points = np.mean(X_train, axis=0).reshape((1, *X_train.shape[1:])).repeat(X_expl.shape[0], axis=0)

    elif background == 'optimized':  # optimized input in vicinity of the anomaly that the network predicts as benign
        if predict_fn is None or device is None:
            raise ValueError("background 'optimized' requires predict_fn and device as input")
        from xai.automated_background_torch import optimize_local_input_gradient_descent
        reference_points = np.zeros(X_expl.shape)
        for i in range(X_expl.shape[0]):
            reference_points[i] = optimize_local_input_gradient_descent(data_point=X_expl[i].reshape((1, *X_expl.shape[1:])),
                                                                        mask=np.zeros((1, *X_expl.shape[1:])),
                                                                        predict_fn=functools.partial(predict_fn, output_cpu=False),
                                                                        device=device)

    else:
        reference_points = None

    return reference_points


def tabular_reference_points(background, X_expl, X_train=None, predict_fn=None):

    if background in ['mean', 'NN']:
        assert X_train is not None, f"background '{background}' requires train data as input at variable 'X_train'"

    if background in ['optimized']:
        assert predict_fn is not None, f"background '{background}' requires predict_fn as input"

    if background == 'zeros':  # zero vector, default
        reference_points = np.zeros(X_expl.shape)
        return reference_points

    elif background == 'mean':  # mean training data point for each data point
        reference_points = np.mean(X_train, axis=0).reshape((1, -1)).repeat(X_expl.shape[0], axis=0)
        return reference_points

    elif background == 'NN':  # nearest neighbor in the normal training data for each data point
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_train)
        neighbor_dist, neighbor_idx = nbrs.kneighbors(X=X_expl, n_neighbors=1, return_distance=True)
        reference_points = X_train[neighbor_idx.flatten()]
        return reference_points

    elif background == 'optimized':  # one normal point in the proximity for each data point
        from xai.automated_background_torch import optimize_input_quasi_newton
        reference_points = np.zeros(X_expl.shape)
        for i in tqdm.tqdm(range(X_expl.shape[0]), desc='generating reference points'):
            reference_points[i] = optimize_input_quasi_newton(data_point=X_expl[i].reshape((1, -1)),
                                                              kept_feature_idx=None,
                                                              predict_fn=predict_fn)
        return reference_points

    else:
        raise ValueError(f"Unkown background: {background}")
