class NegativeSampler:
    """
    Given an input DGL HeteroGraph and a given edge type, this class generates
    negative samples for that edge type with a ratio of 1:1. """

    def __init__(self, graph, edge_type):
        self.graph = graph
        self.edge_type = edge_type
        self.negative_edges = None

    def sample(self):
        import dgl
        import torch

        src_nodes, dst_nodes = self.graph.edges(etype=self.edge_type)
        num_edges = len(src_nodes)

        # Randomly sample negative edges
        neg_src_nodes = torch.randint(0, self.graph.num_nodes(self.edge_type[0]), (num_edges,))
        neg_dst_nodes = torch.randint(0, self.graph.num_nodes(self.edge_type[2]), (num_edges,))

        # Create negative edges
        self.negative_edges = (neg_src_nodes, neg_dst_nodes)

        return dgl.heterograph({self.edge_type: self.negative_edges})