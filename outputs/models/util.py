import pickle

from anomaly_detection.autoencoder_torch import Autoencoder


def load_best_detector(model):
    if model == 'AE':
            detector = Autoencoder(**pickle.load(open('./outputs/models/AE_session2.p', 'rb')))
            detector = detector.load(f'outputs/models/AE_session2_torch')
    else:
        raise ValueError(f"Expected 'model' to be 'AE', but was {model}")
    return detector
