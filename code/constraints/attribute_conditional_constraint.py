import torch
from gurobipy import GRB, abs_, and_, or_

from constraints import AbstractConstraint
from dl2.dl2lib import GT, LT, And, Implication, Or
from dl2.training.supervised.domains import CategoricalBox


class AttributeConditionalConstraint(AbstractConstraint):

    def __init__(self, model, delta, below_epsilon, above_epsilon, att_col_idx, att_threshold, col_means, col_stds, continuous_cols):
        super().__init__(model)

        self.model = model
        self.delta = delta
        self.below_epsilon = below_epsilon
        self.above_epsilon = above_epsilon

        self.att_col_idx = att_col_idx
        self.att_threshold = att_threshold
        self.col_means = col_means
        self.col_stds = col_stds
        self.continuous_cols = continuous_cols
        self.cat = {}

        self.norm_threshold = (self.att_threshold / self.col_stds[self.att_col_idx]).item()

        self.n_tvars = 1
        self.n_gvars = 1

    def get_domains(self, x_batches, _):
        assert len(x_batches) == 1
        batch_size, num_features = x_batches[0].shape

        epsilon = torch.zeros(1, num_features).to(x_batches[0].device)
        epsilon[0, self.continuous_cols] = max(self.below_epsilon, self.above_epsilon)
        lb = x_batches[0] - epsilon
        ub = x_batches[0] + epsilon

        return [CategoricalBox(lb, ub, self.cat)]

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        data_orig = x_batches[0]
        data_adv = z_inp[0]

        att_orig, att_adv = data_orig[:, self.att_col_idx], data_adv[:, self.att_col_idx]
        abs_att_diff = (att_orig - att_adv).abs()

        f1 = And([LT(att_orig, self.norm_threshold), LT(att_adv, self.norm_threshold), LT(abs_att_diff, self.below_epsilon)])
        f2 = And([GT(att_orig, self.norm_threshold), GT(att_adv, self.norm_threshold), LT(abs_att_diff, self.above_epsilon)])

        pre_condition = Or([f1, f2])

        latent_data = self.model.encode(data_orig)
        latent_adv = self.model.encode(data_adv)
        l_inf = torch.abs(latent_data - latent_adv).max(1)[0]
        post_condition = LT(l_inf, self.delta)

        return Implication(pre_condition, post_condition)

    def lt_ineq(self, a, b, grb_model):
        z = grb_model.addVar(vtype=GRB.BINARY)
        grb_model.addConstr((z == 1) >> (a <= b))
        grb_model.addConstr((z == 0) >> (b <= a))

        return z

    def get_grb_vars(self, grb_model, x_batches, y_batches):
        domains = self.get_domains(x_batches, y_batches)[0]
        x_inp = domains.get_grb_vars(grb_model)
        grb_model.update()

        data_orig = x_batches[0]
        att_orig = data_orig[0, self.att_col_idx].item()
        att_adv = x_inp[self.att_col_idx]

        att_diff = grb_model.addVar(-GRB.INFINITY, GRB.INFINITY)
        grb_model.addConstr(att_diff == (att_orig - att_adv))
        abs_att_diff = grb_model.addVar(0, GRB.INFINITY)
        grb_model.addConstr(att_diff == abs_(abs_att_diff))

        f1 = grb_model.addVar(vtype=GRB.BINARY)
        f2 = grb_model.addVar(vtype=GRB.BINARY)
        if att_orig <= self.norm_threshold:
            grb_model.addConstr(f1 == and_([
                self.lt_ineq(att_adv, self.norm_threshold, grb_model),
                self.lt_ineq(abs_att_diff, self.below_epsilon, grb_model),
            ]))
        if att_orig > self.norm_threshold:
            grb_model.addConstr(f2 == and_([
                self.lt_ineq(att_adv, self.norm_threshold, grb_model),
                self.lt_ineq(abs_att_diff, self.above_epsilon, grb_model),
            ]))

        pre_condition = grb_model.addVar(vtype=GRB.BINARY)
        grb_model.addConstr(pre_condition == or_([f1, f2]))
        grb_model.addConstr(pre_condition == 1)

        return x_inp
