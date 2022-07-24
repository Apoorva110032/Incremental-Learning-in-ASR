import torch
import torch.nn as nn

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LossType, NeuralType, LogprobsType, IntType, AcousticEncodedRepresentation


__all__ = ['EBKDLoss']


class EBKDLoss(Serialization, Typing, nn.CTCLoss):
    @property
    def input_types(self):
        """Input types definitions for EBKDLoss.
        """
        return {
            "teacher_feature_map": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "student_feature_map": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "teacher_importance_map": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "student_importance_map": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())
        }
    
    @property
    def output_types(self):
        """Output types definitions for EBKDLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}
    
    def _init_(self, num_classes, zero_infinity=False, reduction='mean_batch'):
        self._blank = num_classes
        # Don't forget to properly call base constructor
        if reduction == 'mean_batch':
            ctc_reduction = 'none'
            self._apply_batch_mean = True
        elif reduction in ['sum', 'mean', 'none']:
            ctc_reduction = reduction
            self._apply_batch_mean = False
        super().__init__(
            blank=self._blank,
            reduction=ctc_reduction,
            zero_infinity=zero_infinity
        )
    
    @typecheck()
    def forward(self, teacher_feature_map, student_feature_map, teacher_importance_map, student_importance_map):
        #         EBKD Loss
        # Getting attention maps
        teacher_attention_map = torch.nn.functional.relu(torch.mul(teacher_importance_map, teacher_feature_map))
        student_attention_map = torch.nn.functional.relu(torch.mul(student_importance_map, student_feature_map))
        
        # Getting normalized vectors
        teacher_norm = torch.norm(teacher_attention_map, dim=2)
        student_norm = torch.norm(student_attention_map, dim=2)
        
        # Clamping elements into a range
        teacher_norm = teacher_norm.clamp(min=1e-5)
        student_norm = student_norm.clamp(min=1e-5)
        
        # Final EBKD Loss Calculation
        ebkd_loss = torch.sum(torch.norm(
            (teacher_attention_map / teacher_norm.unsqueeze(2)) - (student_attention_map / student_norm.unsqueeze(2)),
            dim=2) / teacher_attention_map.shape[1], 1)
        ebkd_loss = torch.mean(ebkd_loss)
        
        return ebkd_loss
