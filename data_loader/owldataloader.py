import os
import pandas as pd
import torch
import dgl
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

class OwlDataLoader:
    """
    DataLoader that reads .tsv and .owl files from a folder,
    extracts gene_disease_associations, ontology assertions, and subclass edges,
    caches intermediate data to a .pt file, and builds a DGL Heterograph with numeric node IDs.

    If use_pstatement_sampler or use_nstatement_sampler is True, the corresponding
    statement edges are extracted into `state_list` and removed from the graph.
    """
    CACHE_NAME = 'data_cache.pt'

    def __init__(self, folder: str,
                 use_pstatement_sampler: bool = False,
                 use_nstatement_sampler: bool = False):
        self.folder = folder
        self.use_pstatement_sampler = use_pstatement_sampler
        self.use_nstatement_sampler = use_nstatement_sampler
        self.state_list = []

        # paths
        cache_path = os.path.join(self.folder, self.CACHE_NAME)

        if os.path.exists(cache_path):
            # load cached data
            cache = torch.load(cache_path)
            self.node2id = cache['node2id']
            self.gda_edges = cache['gda_edges']
            self.gda_negedges = cache['gda_negedges']
            self.pos_statement = cache['pos_statement']
            self.neg_statement = cache['neg_statement']
            self.subclass_edges = cache['subclass_edges']
        else:
            # parse files and build mapping
            self.gda_edges = []
            self.gda_negedges = []
            self.pos_statement = []
            self.neg_statement = []
            self.subclass_edges = []
            self._read_files()
            self._create_node_mapping()
            # save cache
            cache_data = {
                'node2id': self.node2id,
                'gda_edges': self.gda_edges,
                'gda_negedges': self.gda_negedges,
                'pos_statement': self.pos_statement,
                'neg_statement': self.neg_statement,
                'subclass_edges': self.subclass_edges
            }
            torch.save(cache_data, cache_path)

        # build graph from loaded or parsed data
        self.graph = self._build_graph()

        # extract and remove statement edges for samplers
        if self.use_pstatement_sampler and 'pos_statement' in self.graph.etypes:
            src, dst = self.graph.edges(etype='pos_statement')
            self.state_list = list(zip(src.tolist(), dst.tolist()))
            eids = torch.arange(self.graph.num_edges('pos_statement'), device=src.device)
            self.graph = dgl.remove_edges(self.graph, eids, etype='pos_statement')
        elif self.use_nstatement_sampler and 'neg_statement' in self.graph.etypes:
            src, dst = self.graph.edges(etype='neg_statement')
            self.state_list = list(zip(src.tolist(), dst.tolist()))
            eids = torch.arange(self.graph.num_edges('neg_statement'), device=src.device)
            self.graph = dgl.remove_edges(self.graph, eids, etype='neg_statement')

    def _read_files(self):
        for fname in os.listdir(self.folder):
            path = os.path.join(self.folder, fname)
            if not os.path.isfile(path):
                continue

            if fname.lower().endswith('.tsv'):
                df = pd.read_csv(path, sep='\t', header=0)
                subj = df.iloc[:, 0].astype(str)
                obj  = df.iloc[:, 1].astype(str)
                lbl  = df.iloc[:, -1]
                for s, o, l in zip(subj, obj, lbl):
                    if l == 1:
                        self.gda_edges.append((s, o))
                    else:
                        self.gda_negedges.append((s, o))

            elif fname.lower().endswith('.owl'):
                g = Graph()
                g.parse(path, format='xml')
                # PositivePropertyAssertion
                PROPERTY_ASSERTION = URIRef("http://www.w3.org/2002/07/owl#PropertyAssertion")
                for assertion in g.subjects(RDF.type, PROPERTY_ASSERTION):
                    for src in g.objects(assertion, OWL.sourceIndividual):
                        for tgt in g.objects(assertion, OWL.targetIndividual):
                            self.pos_statement.append((str(src), str(tgt)))
                # NegativePropertyAssertion
                for assertion in g.subjects(RDF.type, OWL.NegativePropertyAssertion):
                    for src in g.objects(assertion, OWL.sourceIndividual):
                        for tgt in g.objects(assertion, OWL.targetIndividual):
                            self.neg_statement.append((str(src), str(tgt)))
                # subClassOf link
                for s, o in g.subject_objects(RDFS.subClassOf):
                    if (s, OWL.deprecated, Literal(True, datatype=XSD.boolean)) in g:
                        continue
                    if (o, OWL.deprecated, Literal(True, datatype=XSD.boolean)) in g:
                        continue
                    self.subclass_edges.append((str(s), str(o)))

    def _create_node_mapping(self):
        nodes = set()
        for edge_list in [self.gda_edges, self.gda_negedges, self.pos_statement,
                          self.neg_statement, self.subclass_edges]:
            for s, t in edge_list:
                nodes.add(s)
                nodes.add(t)
        self.node2id = {uri: idx for idx, uri in enumerate(sorted(nodes))}


    def _build_graph(self) -> dgl.DGLHeteroGraph:
        def to_tensor(pairs):
            src = [self.node2id[s] for s, _ in pairs]
            dst = [self.node2id[t] for _, t in pairs]
            return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)

        data_dict = {}
        if self.gda_edges:
            data_dict[('node', 'gene_disease_association', 'node')] = to_tensor(self.gda_edges)
        if self.pos_statement:
            data_dict[('node', 'pos_statement', 'node')] = to_tensor(self.pos_statement)
        if self.neg_statement:
            data_dict[('node', 'neg_statement', 'node')] = to_tensor(self.neg_statement)
        if self.subclass_edges:
            data_dict[('node', 'link', 'node')] = to_tensor(self.subclass_edges)

        g = dgl.heterograph(data_dict)
        num_nodes = g.num_nodes('node')
        g.nodes['node'].data['feat'] = torch.randn(num_nodes, 128)
        return g

    def get_graph(self) -> dgl.DGLHeteroGraph:
        return self.graph

    def get_uri_mapping(self) -> dict:
        return self.node2id

    def get_edge_lists(self) -> dict:
        return {
            'gene_disease_association': [(self.node2id[s], self.node2id[t]) for s, t in self.gda_edges],
            'pos_statement': [(self.node2id[s], self.node2id[t]) for s, t in self.pos_statement],
            'neg_statement': [(self.node2id[s], self.node2id[t]) for s, t in self.neg_statement],
            'link': [(self.node2id[s], self.node2id[t]) for s, t in self.subclass_edges],
        }
    
    def get_negative_edges(self) -> torch.Tensor:
        """
        Returns a 2×E tensor of source and target node IDs for gene-disease negative edges.
        """
        if not hasattr(self, 'gda_negedges') or not self.gda_negedges:
            return torch.empty((2,0), dtype=torch.long)
        src_ids = [self.node2id[s] for s, _ in self.gda_negedges]
        dst_ids = [self.node2id[t] for _, t in self.gda_negedges]
        src = torch.tensor(src_ids, dtype=torch.long)
        dst = torch.tensor(dst_ids, dtype=torch.long)
        return torch.stack([src, dst], dim=0)

    def get_state_list(self) -> list:
        """
        Returns the extracted statement edge list (pos_statement or neg_statement) if sampler is used.
        """
        return self.state_list