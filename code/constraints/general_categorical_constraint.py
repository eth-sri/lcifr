import torch

from constraints import AbstractConstraint
from dl2 import dl2lib
from dl2.training.supervised.domains import CategoricalBox

EPS = 1e-4


class GeneralCategoricalConstraint(AbstractConstraint):

    def __init__(self, model, delta, epsilon, cat, continuous_columns):
        super().__init__(model)
        self.delta = delta
        self.epsilon = epsilon + EPS
        self.cat = cat

        self.all_cat_columns = []
        for ids in self.cat.values():
            self.all_cat_columns += ids

        self.continuous_columns = continuous_columns
        self.n_tvars = 1
        self.n_gvars = 1

    def get_domains(self, x_batches, _):
        assert len(x_batches) == 1
        batch_size, num_features = x_batches[0].shape

        epsilon = torch.zeros(1, num_features).to(x_batches[0].device, dtype=x_batches[0].dtype)
        epsilon[0, self.continuous_columns] = self.epsilon
        lb = x_batches[0] - epsilon
        ub = x_batches[0] + epsilon
        lb[:, self.all_cat_columns] = -EPS
        ub[:, self.all_cat_columns] = +EPS

        return [CategoricalBox(lb, ub, self.cat)]

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        data_orig = x_batches[0]
        data_adv = z_inp[0]

        latent_data = self.model.encode(data_orig)
        latent_adv = self.model.encode(data_adv)

        l_inf = torch.abs(latent_data - latent_adv).max(1)[0]

        return dl2lib.LT(l_inf, self.delta)

    def get_grb_vars(self, grb_model, x_batches, y_batches):
        x_inp = self.get_domains(x_batches, y_batches)[0].get_grb_vars(grb_model)

        return x_inp
