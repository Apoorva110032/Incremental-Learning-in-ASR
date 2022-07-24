import torch
import torch.nn as nn

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LossType, NeuralType, LogprobsType, IntType

_all_ = ['RBKDLoss']


class RBKDLoss(Serialization, Typing, nn.CTCLoss):

    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "teacher_logits": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "student_logits": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "temperature": NeuralType(elements_type=IntType())
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_classes, zero_infinity=False, reduction='mean_batch', temperature=3):
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
        self.temperature = temperature

    @typecheck
    def forward(self, teacher_logits, student_logits):
        # RBKD Loss
        # teacher soft probability
        p_teacher = torch.div(torch.pow(teacher_logits, 1 / self.temperature),
                              torch.sum(torch.pow(teacher_logits, 1 / self.temperature), 2))
        # student soft probability
        p_student = torch.div(torch.pow(student_logits, 1 / self.temperature),
                              torch.sum(torch.pow(student_logits, 1 / self.temperature), 2))
        # loss
        rbkd_loss = torch.mean(torch.sum(torch.sum(-1 * (p_teacher * (torch.log(p_student))), 2), 1))

        return rbkd_loss
