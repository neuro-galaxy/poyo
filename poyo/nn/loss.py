import torch
import torch.nn.functional as F

from torchmetrics import R2Score

from poyo.taxonomy import OutputType


def compute_loss_or_metric(
    loss_or_metric: str,
    output_type: OutputType,
    output: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    r"""Helper function to compute various losses or metrics for a given output type.

    It supports both continuous and discrete output types, and a variety of losses
    and metrics, including mse loss, binary cross entropy loss, and R2 score.

    Args:
        loss_or_metric: The name of the metric to compute.
        output_type: The nature of the output. One of the values from OutputType.
        output: The output tensor.
        target: The target tensor.
        weights: The sample-wise weights for the loss computation.
    """
    if output_type == OutputType.CONTINUOUS:
        if loss_or_metric == "mse":
            # MSE loss
            loss_noreduce = F.mse_loss(output, target, reduction="none").mean(dim=1)
            return (weights * loss_noreduce).sum() / weights.sum()
        elif loss_or_metric == "r2":
            r2score = R2Score(num_outputs=target.shape[1])
            return r2score(output, target)
        else:
            raise NotImplementedError(
                f"Loss/Metric {loss_or_metric} not implemented for continuous output"
            )

    if output_type in [
        OutputType.BINARY,
        OutputType.MULTINOMIAL,
        OutputType.MULTILABEL,
    ]:
        if loss_or_metric == "bce":
            target = target.squeeze(dim=1)
            loss_noreduce = F.cross_entropy(output, target, reduction="none")
            if loss_noreduce.ndim > 1:
                loss_noreduce = loss_noreduce.mean(dim=1)
            return (weights * loss_noreduce).sum() / weights.sum()
        elif loss_or_metric == "accuracy":
            pred_class = torch.argmax(output, dim=1)
            return (pred_class == target.squeeze()).sum() / len(target)
        else:
            raise NotImplementedError(
                f"Loss/Metric {loss_or_metric} not implemented for binary/multilabel "
                "output"
            )

    raise NotImplementedError(
        "I don't know how to handle this task type. Implement plis"
    )
