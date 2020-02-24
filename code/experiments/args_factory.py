import argparse

from dl2 import dl2lib


def get_args():
    parser = argparse.ArgumentParser()

    dl2lib.add_default_parser_args(parser)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--encoder-layers', type=int, nargs='+', required=True)
    parser.add_argument('--decoder-layers', type=int, nargs='+', required=True)
    parser.add_argument('--constraint', type=str, required=True)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--dec-weight', type=float, default=0.0)
    parser.add_argument('--dl2-weight', type=float, default=0.0)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--models-base', type=str, default='')
    parser.add_argument('--dl2-iters', type=int, default=25)
    parser.add_argument('--dl2-lr', type=float, default=0.05)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--quantiles', action='store_true')
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--load-epoch', type=int, default=None)
    parser.add_argument('--protected-att', type=str, default=None)
    parser.add_argument('--adversarial', action='store_true')
    parser.add_argument('--delta', type=float, default=None)

    args = parser.parse_args()

    if args.adversarial and args.delta is None:
        parser.error('--adversarial and --delta must be given together')

    return args
