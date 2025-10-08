#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import json
from typing import Any

import torch
import torch.nn as nn
from timm.models import create_model

from .mci import GlobalPool2D


def load_model_config(
        model_name: str,
) -> Any:
    # Strip suffixes to model name
    model_name = "_".join(model_name.split("_")[0:2])

    # Config files
    root_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(root_dir, "configs")
    model_cfg_file = os.path.join(configs_dir, model_name + ".json")

    # Get config from yaml file
    if not os.path.exists(model_cfg_file):
        raise ValueError(f"Unsupported model name: {model_name}")
    model_cfg = json.load(open(model_cfg_file, "r"))

    return model_cfg


class MCi(nn.Module):
    """
    This class implements `MCi Models <https://arxiv.org/pdf/2311.17049.pdf>`_
    """

    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super().__init__()
        self.projection_dim = None
        if "projection_dim" in kwargs:
            self.projection_dim = kwargs.get("projection_dim")
            # Remove projection_dim from kwargs to avoid passing it to create_model
            kwargs = {k: v for k, v in kwargs.items() if k != "projection_dim"}

        # Create model without projection_dim
        self.model = create_model(model_name, **kwargs)

        # Build out projection head.
        if self.projection_dim is not None:
            if hasattr(self.model, "head"):
                self.model.head = MCi._update_image_classifier(
                    image_classifier=self.model.head, projection_dim=self.projection_dim
                )

    def forward(self, x: Any, *args, **kwargs) -> Any:
        """A forward function of the model."""
        # Check if return_image_embeddings is requested
        return_image_embeddings = kwargs.pop('return_image_embeddings', False)
        
        # Forward through the base model
        x = self.model(x, *args, **kwargs)
        
        # If return_image_embeddings is requested, return a dict with both logits and embeddings
        if return_image_embeddings:
            # We need to get the intermediate features before the classification head
            # Hook into the model to get the conv_exp output
            if hasattr(self.model, 'conv_exp'):
                # Forward through the network to get intermediate features
                with torch.no_grad():
                    # Get embeddings from conv_exp layer (before classification head)
                    # Return whatever format the model naturally produces
                    embeddings = self.model.conv_exp(self.model.forward_tokens(self.model.forward_embeddings(x)))
                return {
                    "logits": x,
                    "image_embeddings": embeddings
                }
            else:
                # Fallback: return logits as embeddings
                return {
                    "logits": x,
                    "image_embeddings": x
                }
        else:
            return x

    @staticmethod
    def _get_in_feature_dimension(image_classifier: nn.Module) -> int:
        """Return the input feature dimension to the image classification head."""
        in_features = None
        
        # Handle ClassifierHead from newer timm versions
        if hasattr(image_classifier, 'fc') and isinstance(image_classifier.fc, nn.Linear):
            in_features = image_classifier.fc.in_features
        elif isinstance(image_classifier, nn.Sequential):
            # Classifier that uses nn.Sequential usually has global pooling and
            # multiple linear layers. Find the first linear layer and get its
            # in_features
            for layer in image_classifier:
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
        elif isinstance(image_classifier, nn.Linear):
            in_features = image_classifier.in_features

        if in_features is None:
            raise NotImplementedError(
                f"Cannot get input feature dimension of {image_classifier}."
            )
        return in_features

    @staticmethod
    def _update_image_classifier(
        image_classifier: nn.Module, projection_dim: int, *args, **kwargs
    ) -> nn.Module:
        in_features = MCi._get_in_feature_dimension(image_classifier)
        new_img_classifier = GlobalPool2D(in_dim=in_features, out_dim=projection_dim)
        return new_img_classifier
