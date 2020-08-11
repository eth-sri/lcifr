import argparse
import random
import time
from os import path, makedirs

import numpy as np
import torch.nn as nn
import torch.utils.data
from gurobipy import GRB, LinExpr, Model
from sklearn.metrics import balanced_accuracy_score

from constraints import ConstraintBuilder
from dl2 import dl2lib
from dl2.training.supervised.oracles import DL2_Oracle
from models import Autoencoder, LogisticRegression

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

DL2_ITERS = 10
DL2_LR = 0.005
EPS = 1e-4


def add_relu_constraints(model, in_lb, in_ub, in_neuron, out_neuron,
                         is_binary):
    if in_ub <= 0:
        out_neuron.lb = 0
        out_neuron.ub = 0

    elif in_lb >= 0:
        model.addConstr(in_neuron, GRB.EQUAL, out_neuron)

    else:
        model.addConstr(out_neuron >= 0)
        model.addConstr(out_neuron >= in_neuron)

        if is_binary:
            relu_ind = model.addVar(vtype=GRB.BINARY)
            model.addConstr(out_neuron <= in_ub * relu_ind)
            model.addConstr(out_neuron <= in_neuron - in_lb * (1 - relu_ind))
            model.addGenConstrIndicator(
                relu_ind, True, in_neuron, GRB.GREATER_EQUAL, 0.0
            )
            model.addGenConstrIndicator(
                relu_ind, False, in_neuron, GRB.LESS_EQUAL, 0.0
            )

        else:
            model.addConstr(
                -in_ub * in_neuron + (in_ub - in_lb) * out_neuron,
                GRB.LESS_EQUAL, -in_lb * in_ub
            )


def propagate(x, layers, grb_model, complete):
    n_outs = len(x[-1])

    for layer_idx, layer in enumerate(layers):
        x[layer_idx] = []

        if isinstance(layer, nn.Linear):
            for i in range(layer.out_features):
                expr = LinExpr()
                expr += layer.bias[i]
                expr += LinExpr(
                    layer.weight[i].detach().cpu().numpy().tolist(),
                    x[layer_idx - 1]
                )
                grb_model.update()

                grb_model.setObjective(expr, GRB.MINIMIZE)
                grb_model.optimize()
                lb = grb_model.objVal - EPS

                grb_model.setObjective(expr, GRB.MAXIMIZE)
                grb_model.optimize()
                ub = grb_model.objVal + EPS

                x[layer_idx] += [
                    grb_model.addVar(
                        lb=lb, ub=ub, vtype=GRB.CONTINUOUS,
                        name='x_{}_{}'.format(layer_idx, i)
                    )
                ]
                grb_model.addConstr(expr, GRB.EQUAL, x[layer_idx][i])
                n_outs = layer.out_features

        elif isinstance(layer, nn.ReLU):
            for i in range(n_outs):
                in_lb, in_ub = x[layer_idx - 1][i].lb, x[layer_idx - 1][i].ub
                x[layer_idx] += [
                    grb_model.addVar(in_lb, in_ub, vtype=GRB.CONTINUOUS)
                ]
                add_relu_constraints(
                    grb_model, in_lb, in_ub, x[layer_idx - 1][i],
                    x[layer_idx][i], complete
                )

        else:
            assert False

        grb_model.update()

    return x, n_outs


parser = argparse.ArgumentParser()
dl2lib.add_default_parser_args(parser)
parser.add_argument('--models-dir', type=str, required=True)
parser.add_argument('--num-certify', type=int, default=None)
parser.add_argument('--complete', action='store_true')
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--load', action='store_true')
parser.add_argument('--label', type=str, default=None)
parser.add_argument('--protected-att', type=str, default=None)
parser.add_argument('--adversarial', action='store_true')
parser.add_argument('--transfer', action='store_true')
parser.add_argument('--quantiles', action='store_true')
args = parser.parse_args()

dataset, constraint, layers = args.models_dir.split('models/')[1].split('/')[:3]
layers = [int(layer) for layer in layers.split('_')]
encoder_layers = layers[:(len(layers) + 1) // 2]
decoder_layers = layers[(len(layers) - 1) // 2:]

project_root = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
results_file = path.join(
    project_root, 'results', args.models_dir.split('models/')[1],
    args.label if args.label else '', ('robust_' if args.adversarial else '') +
    'complete.txt' if args.complete else 'incomplete.txt'
)
makedirs(path.dirname(results_file), exist_ok=True)
open(results_file, 'w')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = getattr(__import__('datasets'), dataset.capitalize() + 'Dataset')
train_dataset = dataset('train', args)
test_dataset = dataset('test', args)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False
)

autoencoder = Autoencoder(encoder_layers, decoder_layers)
classifier = LogisticRegression(encoder_layers[-1])

for param in autoencoder.parameters():
    param.data = param.double()
for param in classifier.parameters():
    param.data = param.double()

prefix = f'{args.label}_' if args.label else ''
postfix = '_robust' if args.adversarial else ''

autoencoder.load_state_dict(torch.load(
    path.join(
        args.models_dir, prefix + f'autoencoder_{args.epoch}' + postfix + '.pt'
    ), map_location=lambda storage, loc: storage
))
classifier.load_state_dict(torch.load(
    path.join(
        args.models_dir, prefix + f'classifier_{args.epoch}' + postfix + '.pt'
    ), map_location=lambda storage, loc: storage
))

encoder_layers = []

for layer in autoencoder.encoder_layers:
    if isinstance(layer, nn.Sequential):
        for sub_layer in layer:
            encoder_layers += [sub_layer]
    else:
        encoder_layers += [layer]

all_l_inf, all_times = [], []
ver, corr = 0, 0

oracle = DL2_Oracle(
    learning_rate=DL2_LR, net=autoencoder, use_cuda=torch.cuda.is_available(),
    constraint=ConstraintBuilder.build(autoencoder, train_dataset, constraint)
)

autoencoder.eval()
classifier.eval()

predictions = list()
labels = list()

for idx, (data_batch, targets_batch, _) in enumerate(test_loader):
    if args.num_certify is not None and idx >= args.num_certify:
        break

    time_start = time.time()

    data_batch = data_batch.double()
    latent_data = autoencoder.encode(data_batch)

    y_pred = classifier.predict(latent_data).detach()
    predictions.append(y_pred.detach().cpu().unsqueeze(0))
    labels.append(targets_batch.detach().cpu())

    if y_pred == targets_batch[0]:
        corr += 1

    x_batches, y_batches = list(), list()
    k = 1

    for i in range(oracle.constraint.n_tvars):
        x_batches.append(data_batch[i: i + k])
        y_batches.append(targets_batch[i: i + k])

    if oracle.constraint.n_gvars > 0:
        domains = oracle.constraint.get_domains(x_batches, y_batches)
        z_batches = oracle.general_attack(
            x_batches, y_batches, domains, num_restarts=1, num_iters=DL2_ITERS,
            args=args
        )
    else:
        z_batches = None

    latent_adv = autoencoder.encode(z_batches[0])
    dl2_loss = oracle.evaluate(x_batches, y_batches, z_batches, args)[1]

    grb_model = Model('milp')
    grb_model.setParam('OutputFlag', False)
    grb_model.setParam('MIPGap', 0)
    grb_model.setParam('NumericFocus', 2)
    grb_model.setParam('FeasibilityTol', 1e-9)
    grb_model.setParam('IntFeasTol', 1e-9)

    x_inp = oracle.constraint.get_grb_vars(
        grb_model, [data_batch], [targets_batch]
    )
    x_data = {-1: x_inp}

    grb_model.update()
    last_layer_idx = len(encoder_layers) - 1
    l_inf = 0

    x_data, n_outs = propagate(
        x_data, encoder_layers, grb_model, args.complete
    )
    grb_model.update()

    for i in range(n_outs):
        l_inf = max(
            l_inf, latent_data[0, i].item() - x_data[last_layer_idx][i].lb
        )
        l_inf = max(
            l_inf, x_data[last_layer_idx][i].ub - latent_data[0, i].item()
        )

    z_data = {
        -1: [
            grb_model.addVar(
                latent_data[0, i].item() - l_inf,
                latent_data[0, i].item() + l_inf, name=f'z_-1_{i}'
            ) for i in range(n_outs)
        ]
    }
    z_data, n_outs = propagate(
        z_data, [classifier.linear], grb_model, args.complete
    )

    grb_model.setObjective(z_data[0][0], GRB.MINIMIZE)
    grb_model.optimize()
    min_logit = grb_model.objVal
    grb_model.setObjective(z_data[0][0], GRB.MAXIMIZE)
    grb_model.optimize()
    max_logit = grb_model.objVal

    out_class = -1
    if min_logit > 0:
        out_class = 1
    elif max_logit < 0:
        out_class = 0

    if out_class == y_pred:
        ver += 1

    time_end = time.time()
    all_l_inf += [l_inf]
    all_times += [time_end - time_start]

    balanced_accuracy = balanced_accuracy_score(
        torch.cat(labels), torch.cat(predictions)
    )

    if len(all_l_inf) % 10 == 0:
        with open(results_file, 'a') as f:
            f.write(
                f'[n={len(all_l_inf):d}] avg time: {np.mean(all_times):.4f} s, '
                f'certified: {ver / (idx + 1):.4f}, '
                f'accuracy: {corr / (idx + 1):.4f}, '
                f'balanced accuracy: {balanced_accuracy:.4f}\n'
            )

