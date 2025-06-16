
class Graph:
    """
    Receives a dict of torch tensors where keys are edge types and values are tensors of shape (2, num_edges).
    Each tensor contains pairs of node indices representing edges in the graph.
    The graph is directed, meaning that each edge has a direction from source to target.
    The output is a DGL HeteroGraph. The node features are randomly initialized.
    """

    def __init__(self, edge_dict):
        import dgl
        import torch

        self.edge_dict = edge_dict
        self.graph = dgl.heterograph(edge_dict)
        self.node_features = {ntype: torch.randn(self.graph.num_nodes(ntype), 128) for ntype in self.graph.ntypes}
        self.graph.ndata['feat'] = self.node_features

    def get_graph(self):
        return self.graph
