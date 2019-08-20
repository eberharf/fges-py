import numpy as np

def generate_graph(num_variables, sparsity=0.1, edge_weight_range=(0.5, 1.5)):
    '''
    Generate a random DAG (with edge weights), represented as a matrix.
    :param num_variables: the number of variables in the graph
    :param sparsity: the sparsity of the edges (i.e. the percentage of edges present compared to a complete DAG)
    :param edge_weight_range: range to sample uniformly to generate edge weights
    :return: the matrix of edge weights, and a topological ordering of the vertices
    '''
    edge_weights = np.random.uniform(edge_weight_range[0], edge_weight_range[1], (num_variables, num_variables))
    # Edge weights are uniformly distributed over +/- edge_weight_range
    edge_weights *= np.random.choice([1, -1], (num_variables, num_variables))
    # Apply sparsity filter
    edge_weights *= (np.random.random((num_variables, num_variables)) < sparsity)
    # Make a DAG
    edge_weights = np.triu(edge_weights, 1)

    # Shuffle vertices
    vertex_shuffle = np.random.permutation(num_variables)
    edge_weights = edge_weights[vertex_shuffle, :]
    edge_weights = edge_weights[:, vertex_shuffle]

    return edge_weights, vertex_shuffle

def generate_data(graph, vertex_order, num_data_points, variable_noise=1):
    num_vertices = graph.shape[0]

    data = np.zeros((num_data_points, num_vertices))

    data[:, vertex_order[0]] = np.random.normal(0, variable_noise, num_data_points)

    for i in range(1, num_vertices):
        data[:, vertex_order[i]] = np.matmul(data[:, vertex_order[:i]], graph[vertex_order[:i], vertex_order[i]])
        data[:, vertex_order[i]] += np.random.normal(0, variable_noise, num_data_points)
    return data