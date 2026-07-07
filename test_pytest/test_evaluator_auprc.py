import torch

from timesead.evaluation.evaluator import Evaluator


def test_auprc_and_ts_auprc_return_floats_when_no_positive_labels():
    labels = torch.zeros(10, dtype=torch.long)
    scores = torch.linspace(0.0, 1.0, steps=10, dtype=torch.float)

    evaluator = Evaluator()

    auprc, _ = evaluator.auprc(labels, scores)
    ts_auprc, _ = evaluator.ts_auprc(labels, scores)

    assert isinstance(auprc, float)
    assert isinstance(ts_auprc, float)
