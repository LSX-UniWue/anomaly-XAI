import torchvision.transforms.transforms
import tqdm
import numpy as np
import pandas as pd
import torch
from captum.attr import DeepLift, IntegratedGradients, LRP, Saliency, InputXGradient
from captum.attr._utils import lrp_rules


def captum_attribute_features(algorithm, input, net, target, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input, target=target, **kwargs)
    return tensor_attributions


def add_lrp_layer_rules(module):
    """Sets LRP backpropagation rules for each type of layer: rules are available at captum.attr._utils"""
    # Kauffmann2020 settings
    if isinstance(module, torch.nn.Linear):
        module.rule = lrp_rules.GammaRule()  # after Kauffmann2020 for linear+relu feature layers
    elif isinstance(module, torch.nn.Conv2d):
        module.rule = lrp_rules.EpsilonRule()  # linear dependency -> same as linear layers
    elif isinstance(module, torch.nn.BatchNorm2d):
        module.rule = lrp_rules.GammaRule()  # linear dependency -> same as linear layers
    elif isinstance(module, torch.nn.ReLU):
        module.rule = lrp_rules.GammaRule()  # after Kauffmann2020 for linear+relu feature layers
    elif isinstance(module, torch.nn.MaxPool2d):
        module.rule = lrp_rules.GammaRule()  # just like ReLUs
    elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
        module.rule = lrp_rules.EpsilonRule()  # after Kauffmann2020 (c_k/N * a_j) since grad*input = 1 * input = 1 * a_j and all layers are normalized
    elif isinstance(module, torch.nn.PairwiseDistance):
        module.rule = PairwiseDistanceRule()  # (input1 - input2) ** 2 after Kauffmann2020
    elif isinstance(module, torch.nn.modules.loss.MSELoss):
        module.rule = PairwiseDistanceRule()  # input ** 2 after Kauffmann2020
    elif isinstance(module, torchvision.transforms.transforms.Normalize):
        module.rule = L2NormRule()


def add_rules_to_layers(model):
    """taken from the _get_layers function used by captum_lrp in captum.attr._core.lrp"""
    for layer in model.children():
        if len(list(layer.children())) == 0:
            add_lrp_layer_rules(layer)
        else:
            add_rules_to_layers(layer)


def explain_anomalies(X_anomalous, model, predict_fn, xai_type, out_template, target=0, reference_points=None, device='cuda'):

    anom_cols, anom_index = None, None
    if isinstance(X_anomalous, pd.DataFrame):
        anom_cols = X_anomalous.columns
        anom_index = X_anomalous.index
        X_anomalous = X_anomalous.values
    if isinstance(reference_points, pd.DataFrame):
        reference_points = reference_points.values

    model = model.to(device)
    model.eval()

    # Initialize XAI
    additional_kwargs = {}
    if xai_type == 'captum_gradient':
        explainer = Saliency(predict_fn)
        additional_kwargs['abs'] = False

    elif xai_type == 'captum_grad_input':
        explainer = InputXGradient(predict_fn)

    elif xai_type == 'captum_deeplift':
        explainer = DeepLift(predict_fn)

    elif xai_type == 'captum_intgrad':
        explainer = IntegratedGradients(predict_fn)

    elif xai_type == 'captum_lrp':
        explainer = LRP(model=model)
        target = None
    else:
        raise ValueError(f'Unknown xai_type: {xai_type}')

    # Calculating attributions
    expl = np.zeros(X_anomalous.shape)  # output explanation
    for sample in tqdm.tqdm(range(X_anomalous.shape[0])):
        input_tensor = torch.Tensor(X_anomalous[sample]).resize(1, *X_anomalous[sample].shape).to(device)
        input_tensor.requires_grad = True

        if xai_type in ['captum_lrp']:  # lrp needs to attach layer rules for every call of attribute
            add_rules_to_layers(model=model)
        elif reference_points is not None and xai_type in ['captum_intgrad']:  # reset reference
            additional_kwargs['baselines'] = torch.Tensor(reference_points[sample]).unsqueeze(0).to(device)

        attr_dl = captum_attribute_features(algorithm=explainer,
                                            input=input_tensor,
                                            net=model,
                                            target=target,
                                            **additional_kwargs)

        expl[sample] = attr_dl.cpu().detach().numpy()
        if xai_type in ['captum_lrp']:  # lrp gives out absolute influence, need to turn negative since our anomaly score is in ]-inf,0]
            expl[sample] *= -1

    if out_template is not None:
        if anom_cols is not None and anom_index is not None:
            pd.DataFrame(expl, columns=anom_cols, index=anom_index).to_csv(out_template.format(xai_type))
        else:
            np.save(file=out_template.format(xai_type), arr=expl)

    return expl


class PairwiseDistanceRule(lrp_rules.EpsilonRule):
    """
    Rule for torch.nn.PairwiseDistance layers after Kauffmann2020, distributing
    second order derivatives for the Euclidean distance.
    $R_{j \leftarrow k} = c_k (a_j - \mu_{jk})^2)$

    [Kauffmann2020. The Clever Hans Effect in Anomaly Detection. arxiv 2020]
    """

    def forward_hook(self, module, inputs, outputs):
        """Registers backward hooks on input and output tensors just like PropagationRule,
        but skips the constant input and gives both inputs to the input hook creation
        so second order can be calculated"""
        inputs = lrp_rules._format_tensor_into_tuples(inputs)
        self._has_single_input = len(inputs) == 1
        self._handle_input_hooks = []
        if not hasattr(inputs[0], "hook_registered"):  # only triggers for the non-constant network prediction
            input_hook = self._create_backward_hook_input(inputs)  # gives both input tensors to the hook creation
            self._handle_input_hooks.append(inputs[0].register_hook(input_hook))
            inputs[0].hook_registered = True
        output_hook = self._create_backward_hook_output(outputs.data)
        self._handle_output_hook = outputs.register_hook(output_hook)
        return outputs.clone()

    def _create_backward_hook_input(self, inputs):
        """Gets both input tensors to compute second order Taylor expansion for distance calculation"""
        def _backward_hook_input(grad):
            relevance = (inputs[0].data - inputs[1].data) ** 2 * grad  # specific second order term * incoming gradient
            device = grad.device
            if self._has_single_input:
                self.relevance_input[device] = relevance.data
            else:
                self.relevance_input[device].append(relevance.data)
            return relevance

        return _backward_hook_input


class L2NormRule(lrp_rules.EpsilonRule):
    """
    Rule for torch.nn.functional.normalize layers analogous to Kauffmann2020,
    distributing second order derivatives for p=2.
    $R_{j \leftarrow k} = c_k (a_j)^2)$

    [Kauffmann2020. The Clever Hans Effect in Anomaly Detection. arxiv 2020]
    """
    def _create_backward_hook_input(self, inputs):
        """Computes second order Taylor expansion for l2 norm calculation"""
        def _backward_hook_input(grad):
            relevance = inputs.data ** 2 * grad  # specific second order term * incoming gradient
            device = grad.device
            if self._has_single_input:
                self.relevance_input[device] = relevance.data
            else:
                self.relevance_input[device].append(relevance.data)
            return relevance

        return _backward_hook_input
