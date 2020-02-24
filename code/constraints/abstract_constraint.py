from abc import ABC, abstractmethod

import torch

from dl2 import dl2lib


class AbstractConstraint(ABC):
    def __init__(self, model):
        self.model = model
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    def eval_z(self, z_batches):
        z_inputs = [z_batch.clone().detach().requires_grad_(True) for z_batch in z_batches]
        z_outputs = [self.model(z_input) for z_input in z_inputs]
        for z_out in z_outputs:
            z_out.requires_grad_(True)
        return z_inputs, z_outputs

    @abstractmethod
    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        pass

    def loss(self, x_batches, y_batches, z_batches, args):
        if z_batches is not None:
            z_inp, z_out = self.eval_z(z_batches)
        else:
            z_inp, z_out = None, None

        constr = self.get_condition(z_inp, z_out, x_batches, y_batches)

        neg_losses = dl2lib.Negate(constr).loss(args)
        pos_losses = constr.loss(args)
        sat = constr.satisfy(args)

        return neg_losses, pos_losses, sat, z_inp
