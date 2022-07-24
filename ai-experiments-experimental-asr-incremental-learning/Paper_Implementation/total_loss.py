import torch
import torch.nn as nn

from typing import Dict, Optional, Union
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from nemo.core.classes import Serialization, Typing, typecheck
from nemo.utils import logging, model_utils
from nemo.core.neural_types import LossType, NeuralType, LogprobsType, LabelsType, LengthsType, AcousticEncodedRepresentation
from nemo.collections.asr.losses.ctc import CTCLoss

from rbkd import RBKDLoss
from ebkd import EBKDLoss

__all__ = ['TotalLoss']


class TotalLoss(CTCLoss):
    
    @property
    def input_types(self):
        """Input types definitions for TotalLoss.
        """
        return {
            "teacher_logits": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "student_logits": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "input_lengths": NeuralType(tuple('B'), LengthsType()),
            "target_lengths": NeuralType(tuple('B'), LengthsType()),
            "teacher_feature_map": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "student_feature_map": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "teacher_importance_map": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "student_importance_map": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
        }
    
    @property
    def output_types(self):
        """Output types definitions for TotalLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}
    
    def __init__(self, *args, **kwargs):
        # Calling Parent Class to access its methods
        super().__init__(
            num_classes=kwargs.get("num_classes"),
            zero_infinity=True,
            reduction=kwargs.get("reduction", "mean_batch")
        )
        
        # Hyperparameters
        self.gamma = kwargs.get("gamma", 100)
        self.beta = kwargs.get("beta", 0.01)
        self.temperature = kwargs.get("temperature", 5)
        
        # CTC Loss Object Creation
        self.CTCLoss = CTCLoss(
            num_classes=kwargs.get("num_classes"),
            zero_infinity=True,
            reduction=kwargs.get("reduction", "mean_batch"),
        )
        
        # RBKD Loss Object Creation
        self.RBKDLoss = RBKDLoss(
            num_classes=kwargs.get("num_classes"),
            zero_infinity=True,
            reduction=kwargs.get("reduction", "mean_batch"),
            temperature=self.temperature
        )
        
        # EBKD Loss Object Creation
        self.EBKDLoss = EBKDLoss(
            zero_infinity=True,
            reduction=kwargs.get("reduction", "mean_batch"),
        )
    
    @typecheck()
    def forward(self, student_logits, targets, input_lengths, target_lengths,
                teacher_logits, teacher_feature_map, student_feature_map, teacher_importance_map, student_importance_map):
        # Calling methods of CTC Loss
        ctc_loss = self.CTCLoss(
            log_probs=student_logits, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths
        )
       
        # Calling methods of RBKD Loss
        rbkd_loss = self.RBKDLoss(
            teacher_logits=teacher_logits, student_logits=student_logits
        )
        
        # Calling methods of EBKD Loss
        ebkd_loss = self.EBKDLoss(
            teacher_feature_map=teacher_feature_map,
            student_feature_map=student_feature_map,
            teacher_importance_map=teacher_importance_map,
            student_importance_map=student_importance_map
        )
        
        # Returning Total Loss
        return ctc_loss + (self.beta * rbkd_loss) + (self.gamma * ebkd_loss)