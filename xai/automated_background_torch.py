
import tqdm
import numpy as np
from scipy.optimize import minimize
import torch


def optimize_local_input_gradient_descent(data_point, mask, predict_fn, max_iter=10000, lr=0.001,
                                          gamma=0.01, flip_channels=False, device='cpu', verbose=1):
    """
    Optimizes given sample for given model function, while keeping certain values in data_point fixed.
    loss = model(y) + gamma * mean squared distance between optimized and original point (excluding fixed values)
    :param data_point:          numpy model input to optimize
    :param mask:                numpy binary mask noting which features should be optimized: 0 means optimize, 1 means keep
    :param predict_fn:          function of pytorch model to optimize loss for
    :param max_iter:            int gradient descent updates
    :param lr:                  float gradient descent learning rate (Adam optimizer)
    :param gamma:               float hyperparameter weighting distance to original point in loss function
    :return:                    numpy optimized data_point
    example:
    x_hat.append(optimize_local_input_gradient_descent(data_point=sample, mask=mask, kept_feature_idx=kept_idx,
                                                       model=model, max_iter=5000, lr=0.001, gamma=0.01))
    """
    if flip_channels:
        data_point = np.transpose(data_point, [0, 3, 1, 2])
        mask = np.transpose(mask, [0, 3, 1, 2])

    orig_point = torch.from_numpy(data_point.astype('float32')).to(device)
    data_point = torch.from_numpy(data_point.astype('float32')).to(device).requires_grad_()
    gamma = torch.Tensor([gamma]).to(device)

    # convert constants to tensors
    mask = torch.from_numpy(mask.astype(bool)).to(device)
    invert_mask = ~mask
    constrained_vals = orig_point * mask

    # optimization procedure
    optimizer = torch.optim.Adam(params=[data_point], lr=lr)

    patience = 10
    delta = 0
    lowest_score = None
    patience_counter = 0
    with tqdm.tqdm(range(max_iter), disable=verbose < 1) as titer:
        for i in titer:
            # forward pass & mask fixed inputs
            y = predict_fn(data_point)
            loss = y + gamma * torch.norm(orig_point - data_point)

            # early stopping
            if lowest_score is None or loss < lowest_score - delta:
                lowest_score = loss
            else:
                patience_counter += 1
            if patience_counter >= patience:
                if flip_channels:
                    return np.transpose(data_point.cpu().detach().numpy(), [0, 2, 3, 1])
                else:
                    return data_point.cpu().detach().numpy()

            # Gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            data_point.data = data_point.data * invert_mask + constrained_vals  # Reset fixed inputs with mask

    if flip_channels:
        return np.transpose(data_point.cpu().detach().numpy(), [0, 2, 3, 1])  # optimized sample
    else:
        return data_point.cpu().detach().numpy()


def optimize_input_quasi_newton(data_point, kept_feature_idx, predict_fn, gamma=0.01, device='cpu'):
    """
    idea from: http://www.bnikolic.co.uk/blog/pytorch/python/2021/02/28/pytorch-scipyminim.html

    Uses quasi-Newton optimization (Sequential Least Squares Programming) to find optimal input alteration for model
    according to:
    loss = -1 * predict_fn(y) + gamma * mean squared distance between optimized and original point (excluding fixed values)
           (optionally + delta * negative distance to points that should be avoided [Haldar2021])
    :param data_point:          numpy model input to optimize
    :param kept_feature_idx:    index of feature in data_point to keep, or None for not constraining any feature
                                Can also contain a list of indices to keep
    :param predict_fn:          function of pytorch model to optimize loss for
    :param gamma:               float factor for adding locality to the optimized input
    :return:                    numpy optimized data_point
    """
    data_point = torch.autograd.Variable(torch.from_numpy(data_point.astype('float32')), requires_grad=True).to(device)
    gamma = torch.Tensor([gamma]).to(device)

    def val_and_grad(x):
        loss = -1 * predict_fn(x) + gamma * torch.linalg.vector_norm(data_point - x)
        loss.backward()
        grad = x.grad
        return loss, grad

    def func(x):
        """scipy needs flattened numpy array with float64, tensorflow tensors with float32"""
        return [vv.cpu().detach().numpy().astype(np.float64).flatten() for vv in
                val_and_grad(torch.tensor(x.reshape([1, -1]), dtype=torch.float32, requires_grad=True))]

    kept_feature_idx = np.where(kept_feature_idx)[0]
    if len(kept_feature_idx) == 0:
        if type(kept_feature_idx) == int:
            constraints = {'type': 'eq', 'fun': lambda x: x[kept_feature_idx] - data_point[:, kept_feature_idx]}
        else:
            from functools import partial
            constraints = []
            for kept_idx in kept_feature_idx:
                constraints.append(
                    {'type': 'eq', 'fun': partial(lambda x, idx: x[idx] - data_point[:, idx], idx=kept_idx)})
    else:
        constraints = ()

    res = minimize(fun=func,
                   x0=data_point.detach().cpu(),
                   jac=True,
                   method='SLSQP',
                   constraints=constraints)
    opt_input = res.x.astype(np.float32).reshape([1, -1])

    return opt_input


def dynamic_synth_data(sample, maskMatrix, model):
    """
    Dynamically generate background "deletion" data for each synthetic data sample by minimizing model output.
    background_type == 'optimized': Finds most benign inputs through SLSQP while always constraining 1 feature,
                                    takes mean of all benign inputs when generating background for samples where
                                    more then 1 feature needs to be constrained (instead of solving SLSQP again).
    :param sample:                  np.ndarray sample to explain, shape (1, n_features)
    :param maskMatrix:              np.ndarray matrix with features to remove in SHAP sampling process
                                    1 := keep, 0 := optimize/remove
    :param model:                   ml-model to optimize loss for
    :return:                        np.ndarray with synthetic samples, shape maskMatrix.shape

    Example1:
    dynamic_synth_data(sample=to_explain[0].reshape([1, -1]),
                    maskMatrix=maskMatrix,
                    model=load_model('../outputs/models/AE_cat'),
                    background_type='full')

    Example2:
    # integrate into SHAP in shap.explainers.kernel @ KernelExplainer.explain(), right before calling self.run()
    if self.dynamic_background:
        from xai.automated_background import dynamic_synth_data
        self.synth_data, self.fnull = dynamic_synth_data(sample=instance.x,
                                                        maskMatrix=self.maskMatrix,
                                                        model=self.full_model,
                                                        background_type=self.dynamic_background)
        self.expected_value = self.fnull
    """
    assert sample.shape[0] == 1, \
        f"Dynamic background implementation can't explain more then one sample at once, but input had shape {sample.shape}"
    assert maskMatrix.shape[1] == sample.shape[1], \
        f"Dynamic background implementation requires sampling of all features (omitted in SHAP when baseline[i] == sample[i]):\n" \
        f"shapes were maskMatrix: {maskMatrix.shape} and sample: {sample.shape}\n" \
        f"Use of np.inf vector as SHAP baseline is recommended"

    # optimize all permutations with 1 kept variable, then aggregate results

    x_hat = []  # contains optimized feature (row) for each leave-one-out combo of varying features (column)
    # Sequential Least Squares Programming
    for kept_idx in tqdm.tqdm(range(sample.shape[1])):
        x_hat.append(optimize_input_quasi_newton(data_point=sample,
                                                 kept_feature_idx=kept_idx,
                                                 predict_fn=model))
    x_hat.append(optimize_input_quasi_newton(data_point=sample,
                                             kept_feature_idx=None,
                                             predict_fn=model))
    x_hat = np.concatenate(x_hat)

    # Find x_tilde by adding x_hat entries for each feature to keep
    def sum_sample(row):
        S = x_hat[:-1][row == True]
        return ((S.sum(axis=0) + x_hat[-1]) / (S.shape[0] + 1)).reshape([1, -1])

    x_tilde_Sc = []
    for mask in maskMatrix:
        x_tilde_Sc.append(sum_sample(mask))
    x_tilde_Sc = np.concatenate(x_tilde_Sc)
    x_tilde = sample.repeat(maskMatrix.shape[0], axis=0) * maskMatrix + x_tilde_Sc * (1 - maskMatrix)

    fnull = model(torch.tensor(x_hat[-1])).unsqueeze(0).detach().numpy()
    return x_tilde, fnull

