def compute_counterfactual(data, sensitive_column, sensitive_values):
    counterfactual = data.clone()
    indices = data[:, sensitive_column] == sensitive_values[0]
    counterfactual[indices, sensitive_column] = sensitive_values[1]
    counterfactual[~indices, sensitive_column] = sensitive_values[0]

    return counterfactual
