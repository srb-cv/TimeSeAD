from ...data.transforms import Transform, WindowTransform
from ..common import AnomalyDetector
from timesead.models.supervised import DSADLoss
import torch
from ...utils import torch_utils
from ...utils.utils import halflife2alpha
from typing import Tuple


class DSADTargetTransform(WindowTransform):
    """Create non-overlapping DSAD windows with labels from the same window."""

    def __init__(
        self,
        parent: Transform,
        window_size: int,
        replace_labels: bool = False,
        reverse: bool = False,
    ) -> None:
        super().__init__(parent, window_size, step_size=window_size, reverse=reverse)
        self.replace_labels = replace_labels


class DSADSupervisionAnomalyDetector(AnomalyDetector):
    def __init__(self, model, criterion: DSADLoss, half_life: int):
        """
        Filonov2016

        :param model:
        :param half_life:
        """
        super(DSADSupervisionAnomalyDetector, self).__init__()

        self.model = model
        self.criterion = criterion
        self.criterion.reduction = 'none'

        self.alpha = halflife2alpha(half_life)

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs) -> None:
        pass

    def compute_online_anomaly_score(
        self,
        inputs: Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], float, float],
    ) -> Tuple[torch.Tensor, float, float]:
        b_inputs, b_targets, moving_avg_num, moving_avg_denom = inputs

        with torch.no_grad():
            representation = self.model(b_inputs)

        window_score = self.criterion((representation,), b_targets).squeeze(-1)

        moving_avg_num, moving_avg_denom = torch_utils.exponential_moving_avg_(
            window_score,
            self.alpha,
            avg_num=moving_avg_num,
            avg_denom=moving_avg_denom,
        )
        _, window_size, _ = b_inputs[0].shape
        return window_score.unsqueeze(1).repeat(1, window_size), moving_avg_num, moving_avg_denom

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

            label = b_targets[0]

            sq_error, moving_avg_num, moving_avg_denom = self.compute_online_anomaly_score((
                b_inputs,
                b_targets,
                moving_avg_num,
                moving_avg_denom))
            errors.append(sq_error)
            labels.append(label.cpu())

        # Supervised DSAD batches are B,W,D in the Hydra flow, so row-major
        # flattening preserves chronological window order.
        scores = torch.cat(errors, dim=0).flatten()
        labels = torch.cat(labels, dim=0).flatten()

        assert labels.shape == scores.shape

        return labels, scores.cpu()
