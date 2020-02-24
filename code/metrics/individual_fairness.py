from sklearn.neighbors import NearestNeighbors


def consistency(features, predictions, num_neighbors):
    nearest_neighbors = NearestNeighbors(num_neighbors, algorithm='ball_tree')
    nearest_neighbors.fit(features)
    neighbors = nearest_neighbors.kneighbors(return_distance=False)

    return 1 - (predictions.unsqueeze(1) - predictions[neighbors]).abs().mean()
