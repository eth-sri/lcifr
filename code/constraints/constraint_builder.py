from constraints import (AttributeConditionalConstraint,
                         GeneralCategoricalConstraint)


class ConstraintBuilder:
    @staticmethod
    def build(model, dataset, constraint):
        constraint, args = constraint.rstrip(')').split('(')
        args = eval(args)

        if constraint == 'GeneralCategorical':
            delta, epsilon, categories = args
            return GeneralCategoricalConstraint(
                model=model,
                delta=delta,
                epsilon=epsilon,
                cat={c: dataset.one_hot_columns[c] for c in categories},
                continuous_columns=dataset.continuous_columns
            )

        elif constraint == 'AttributeConditional':
            delta, below_epsilon, above_epsilon, att, att_threshold = args
            return AttributeConditionalConstraint(
                model=model,
                delta=delta,
                below_epsilon=below_epsilon,
                above_epsilon=above_epsilon,
                att_col_idx=dataset.column_ids[att],
                att_threshold=att_threshold,
                col_means=dataset.mean, col_stds=dataset.std,
                continuous_cols=dataset.continuous_columns
            )

        else:
            raise AttributeError('Unknown constraint')
