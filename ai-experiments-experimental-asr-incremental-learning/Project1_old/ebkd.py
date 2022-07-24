import torch
import torch.nn as nn

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LossType, NeuralType, LogprobsType, IntType

_all_ = ['EBKDLoss']


class EBKDLoss(Serialization, Typing, nn.CTCLoss):
    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "teacher_logits": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "student_logits": NeuralType(('B', 'T', 'D'), LogprobsType())
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def _init_(self, num_classes, zero_infinity=False, reduction='mean_batch'):
        # Don't forget to properly call base constructor
        self._blank = num_classes
        # Don't forget to properly call base constructor
        if reduction == 'mean_batch':
            ctc_reduction = 'none'
            self._apply_batch_mean = True
        elif reduction in ['sum', 'mean', 'none']:
            ctc_reduction = reduction
            self._apply_batch_mean = False
        super().__init__(blank=self._blank, reduction=ctc_reduction, zero_infinity=zero_infinity)

    @typecheck
    def forward(self, teacher_logits, student_logits, teacher_feature_map, student_feature_map):
        # EBKD Loss
        # teacher greedy prediction probability - 1D Tensor
        teacher_gpp = torch.prod(torch.max(teacher_logits, 2), 1)
        # student greedy prediction probability - 1D Tensor
        student_gpp = torch.prod(torch.max(student_logits, 2), 1)

        teacher_importance_map = torch.autograd.grad(torch.log(teacher_gpp), teacher_feature_map)
        student_importance_map = torch.autograd.grad(torch.log(student_gpp), student_feature_map)

        teacher_attention_map = torch.nn.ReLU(torch.mul(teacher_importance_map, teacher_feature_map))
        student_attention_map = torch.nn.ReLU(torch.mul(student_importance_map, student_feature_map))

        # normalized teacher attention map
        teacher_norm = torch.norm(teacher_attention_map, 2)
        # normalized student attention map
        student_norm = torch.norm(student_attention_map, 2)

        # loss
        ebkd_loss = torch.mean(torch.sum(torch.norm(
            (teacher_attention_map / teacher_norm) - (student_attention_map / student_norm)
        ) / teacher_attention_map.shape[1], 1))

        return ebkd_loss
