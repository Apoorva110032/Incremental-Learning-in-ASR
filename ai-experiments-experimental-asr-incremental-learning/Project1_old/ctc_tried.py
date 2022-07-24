import torch
from torch import nn

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType, LogitsType
import torch.nn.functional as F

_all_ = ['CTCLoss']


class CTCLoss(nn.CTCLoss, Serialization, Typing):
    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            # 'y-predicted' --> logits or log probabilities
            "student_logits": NeuralType(('B', 'T', 'D'), LogitsType()),
            # Actual Target Text Transcript in int - Target labels(y): a text transcript of what was spoken - target
            # label sequence (transcription) we want to recognize - associated label sequence - 'y-true'
            "targets": NeuralType(('B', 'T'), LabelsType()),
            # Length of a sample's number of time steps (different to make intervals of equal lengths)
            "input_lengths": NeuralType(tuple('B'), LengthsType()),
            # Length of a sample's target sequence encoding
            "target_lengths": NeuralType(tuple('B'), LengthsType())
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def _init_(self, num_classes: object, zero_infinity: object = False, reduction: object = 'mean_batch') -> object:
        self._blank = num_classes
        # Don't forget to properly call base constructor
        if reduction == 'mean_batch':
            ctc_reduction = 'none'
            self._apply_batch_mean = True
        elif reduction in ['sum', 'mean', 'none']:
            ctc_reduction = reduction
            self._apply_batch_mean = False
        super()._init_(blank=self._blank, reduction=ctc_reduction, zero_infinity=zero_infinity)

    @typecheck()
    def forward(self, student_logits, targets, input_lengths, target_lengths, ctc_path):
        # override forward implementation
        # custom logic, if necessary
        # input_lengths = input_lengths.long()
        # target_lengths = target_lengths.long()
        # targets = targets.long()
        # here we transpose because we expect [B, T, D] while PyTorch assumes [T, B, D]
        student_logits = student_logits.transpose(1, 0)
        # loss = super().forward(
        #     log_probs=log_probs, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths
        # )
        ctc_path = F.one_hot(ctc_path, num_classes=student_logits.shape[2])
        ctc_loss = torch.mean(-1 * torch.log(torch.sum(torch.prod(torch.sum(torch.mul(student_logits, ctc_path), 3), 2),
                                                       1)))

        return ctc_loss