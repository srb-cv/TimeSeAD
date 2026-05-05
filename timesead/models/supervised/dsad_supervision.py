from ...data.transforms import SupervisionTargetTransform, Transform
from ..common import AnomalyDetector
from timesead.models.supervised import DSADLoss
import torch
from ...utils import torch_utils
from typing import Tuple


class DSADTargetTransform(SupervisionTargetTransform):
    def __init__(self, parent: Transform, window_size: int, replace_labels: bool = False,
                 reverse: bool = False):
        super(DSADTargetTransform, self).__init__(parent, window_size, window_size, replace_labels=replace_labels,
                                                     step_size=window_size, reverse=reverse)
        

class DSADSupervisionAnomalyDetector():
    def __init__(self, model: AnomalyDetector, criterion: DSADLoss):
        """
        Filonov2016

        :param model:
        :param half_life:
        """
        super(DSADSupervisionAnomalyDetector, self).__init__()

        self.model = model
        self.criterion = criterion

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs) -> None:
        pass

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, torch.Tensor, float, float]) \
            -> Tuple[torch.Tensor, float, float]:
        # x: (T, B, D), target: (T, B, D), moving_avg: ()
        x, target, moving_avg_num, moving_avg_denom = inputs

        with torch.no_grad():
            x_pred = self.model((x,))

        s = self.criterion(x_pred)
        sq_error = target - x_pred
        torch.square(sq_error, out=sq_error)
        sq_error = torch.sum(sq_error, dim=-1)

        T, B = sq_error.shape
        sq_error = sq_error.T.flatten()
        moving_avg_num, moving_avg_denom = torch_utils.exponential_moving_avg_(sq_error, self.alpha,
                                                                               avg_num=moving_avg_num,
                                                                               avg_denom=moving_avg_denom)

        return sq_error.view(B, T).T, moving_avg_num, moving_avg_denom

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def get_labels_and_scores(self, dataset: torch.utils.data.DataLoader, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        errors = []
        labels = []
        moving_avg_num = 0
        moving_avg_denom = 0

        # Compute exp moving average of error score
        for b_inputs, b_targets in dataset:
            b_inputs = tuple(b_inp.to(self.dummy.device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(self.dummy.device) for b_tar in b_targets)

            x, = b_inputs
            label, target = b_targets

            sq_error, moving_avg_num, moving_avg_denom = self.compute_online_anomaly_score((x, target, moving_avg_num,
                                                                                            moving_avg_denom))
            errors.append(sq_error)
            labels.append(label.cpu())

        scores = torch.cat(errors, dim=1).transpose(0, 1).flatten()
        labels = torch.cat(labels, dim=1).transpose(0, 1).flatten()

        assert labels.shape == scores.shape

        return labels, scores.cpu()