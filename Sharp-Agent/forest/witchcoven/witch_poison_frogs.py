"""Main class, holding information about models and training/testing routines."""

import torch

from ..consts import BENCHMARK
from ..utils import bypass_last_layer

torch.backends.cudnn.benchmark = BENCHMARK

from .witch_base import _Witch


class WitchFrogs(_Witch):
    """Brew poison frogs poison with given arguments.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def _define_objective(
        self, inputs, labels, criterion, sources, target_classes, true_classes
    ):
        """Implement the closure here."""

        def closure(model, optimizer, source_grad, source_clean_grad, source_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            # Carve up the model
            feature_model, last_layer = bypass_last_layer(model)

            # Get standard output:
            outputs = feature_model(inputs)
            outputs_sources = feature_model(sources)
            prediction = (last_layer(outputs).data.argmax(dim=1) == labels).sum()

            feature_loss = (
                (outputs.mean(dim=0, keepdim=True) - outputs_sources).pow(2).mean()
            )
            feature_loss.backward(retain_graph=self.retain)
            return feature_loss.detach().cpu(), prediction.detach().cpu()

        return closure
