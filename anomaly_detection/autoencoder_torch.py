
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


class Autoencoder(torch.nn.Module):
    """
    Autoencoder being build with n_bottleneck neurons in bottleneck.
    Encoder and decoder contain n_layers each.
    size of layers starts at 2**(log2(n_bottleneck) + 1) near bottleneck and increases with 2**(last+1)
    """
    def __init__(self, n_inputs, cpus=0, n_layers=3, n_bottleneck=2**3, seed=0, **params):
        # setting number of threads for parallelization
        super(Autoencoder, self).__init__()

        torch.manual_seed(seed)
        if cpus > 0:
            torch.set_num_threads(cpus * 2)

        bottleneck_exp = (np.log2(n_bottleneck))

        # AE architecture
        layers = []
        # Input
        layers += [torch.nn.Linear(in_features=n_inputs,
                                   out_features=int(2**(bottleneck_exp + n_layers))),
                   torch.nn.ReLU()]
        # Encoder
        for i in range(n_layers - 1, 0, -1):  # layers from bottleneck: 8, 16, 32, 64, ...
            layers += [torch.nn.Linear(in_features=int(2**(bottleneck_exp + i + 1)),
                                       out_features=int(2**(bottleneck_exp + i))),
                       torch.nn.ReLU()]
        # Bottleneck
        layers += [torch.nn.Linear(in_features=int(2**(bottleneck_exp + 1)),
                                   out_features=n_bottleneck)]
        # Decoder
        for i in range(1, n_layers + 1):  # layers from bottleneck: 8, 16, 32, 64, ...
            layers += [torch.nn.Linear(in_features=int(2**(bottleneck_exp + i - 1)),
                                       out_features=int(2**(bottleneck_exp + i))),
                       torch.nn.ReLU()]
        # Output
        layers += [torch.nn.Linear(in_features=int(2**(bottleneck_exp + n_layers)),
                                   out_features=n_inputs)]  # output layer
        # Full model
        self.model = torch.nn.Sequential(*layers)
        self.add_module('distance_layer', module=torch.nn.PairwiseDistance(p=2))

        if 'learning_rate' in params:
            self.optim = torch.optim.Adam(params=self.model.parameters(), lr=params.pop('learning_rate'))
        else:
            self.optim = torch.optim.Adam(params=self.model.parameters())

        self.params = params

    def score_samples(self, x, output_to_numpy=True):
        x = self.to_tensor(x)
        loss = self.__call__(input_tensor=x)
        if output_to_numpy:
            return loss.data.numpy()
        else:
            return loss

    def to_tensor(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        if not torch.is_tensor(x):
            x = torch.Tensor(x)
        return x

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model = self.model.eval()

    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        return self

    def __call__(self, input_tensor, *args, **kwargs):
        pred = self.model(input_tensor)
        loss = -1 * self.distance_layer(pred, input_tensor)  # TODO: inupt_tensor.clone().detach() for LRP to not give it a gradient
        return loss

    def fit(self, data):
        verbose = self.params['verbose']
        dataset = torch.utils.data.TensorDataset(torch.Tensor(data.values), torch.Tensor(data.values))
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=self.params['batch_size'],
                                                  shuffle=self.params['shuffle'])

        for _ in tqdm(range(self.params['epochs']), disable=verbose < 1):
            for x, y in tqdm(data_loader, disable=verbose < 2):

                y_pred = self.model(x)
                loss = self.distance_layer(y_pred, y)

                self.model.zero_grad()
                loss.backward()
                self.optim.step()
        return self
