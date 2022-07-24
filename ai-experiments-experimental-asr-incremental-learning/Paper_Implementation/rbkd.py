import torch
import torch.nn as nn

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LossType, NeuralType, LogprobsType, IntType

__all__ = ['RBKDLoss']


class RBKDLoss(Serialization, Typing, nn.CTCLoss):
    
    @property
    def input_types(self):
        """Input types definitions for RBKDLoss.
        """
        return {
            "teacher_logits": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "student_logits": NeuralType(('B', 'T', 'D'), LogprobsType()),
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
    
    @typecheck()
    def forward(self, teacher_logits, student_logits):
        # RBKD Loss
        
        # here we transpose because we expect [B, T, D] while PyTorch assumes [T, B, D]
        teacher_logits = teacher_logits.transpose(1, 0)
        student_logits = student_logits.transpose(1, 0)
        
        p_teacher = torch.div((torch.exp(teacher_logits / self.temperature)).permute(2, 0, 1),
                              (torch.exp(teacher_logits / self.temperature)).sum(dim=2)).permute(1, 2, 0)
        p_student = torch.div((torch.exp(student_logits / self.temperature)).permute(2, 0, 1),
                              (torch.exp(student_logits / self.temperature)).sum(dim=2)).permute(1, 2, 0)
        
        # Final RBKD Loss Calculation
        rbkd_loss = torch.sum(torch.sum(-1 * (p_teacher * (torch.log(p_student))), 2), 0)
        
        if self._apply_batch_mean:
            rbkd_loss = torch.mean(rbkd_loss)
        
        return rbkd_loss
