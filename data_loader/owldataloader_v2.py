import os, torch, dgl
import pandas as pd
from typing import Union
from rdflib import Graph, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL
from torch_geometric.data import HeteroData



class OwlDataLoader:
    """
    DataLoader that reads .tsv and .owl files from a folder,
    extracts gene_disease_associations, ontology assertions, and subclass edges,
    caches intermediate data to a .pt file, and builds a DGL Heterograph with numeric node IDs.

    If use_pstatement_sampler or use_nstatement_sampler is True, the corresponding
    statement edges are extracted into `state_list` and removed from the graph.
    """
    CACHE_NAME = 'data_cache.pt'

    def __init__(self, folder: str, use_pstatement_sampler: bool = False,
                 use_nstatement_sampler: bool = False, feature_dim: int = 128, graph_type: str = 'dgl'):
        self.folder = folder
        self.use_pstatement_sampler = use_pstatement_sampler
        # self.use_nstatement_sampler = use_nstatement_sampler
        self.state_list = []
        self.feature_dim = feature_dim
        cache_path = os.path.join(self.folder, self.CACHE_NAME)

        if os.path.exists(cache_path):
            cache = torch.load(cache_path)
            self.node2id,self.gda_edges = cache['node2id'], cache['gda_edges'] # self.gda_negedges =  cache['gda_negedges']
            self.pos_statement, self.neg_statement = cache['pos_statement']#, cache['neg_statement']
            self.subclass_edges = cache['subclass_edges']
        else:
            self.gda_edges, self.gda_negedges, self.pos_statement, self.neg_statement, self.subclass_edges = [],[],[],[],[]
            self._read_files()
            self.gda_edges = list(set(self.gda_edges))
            #self.gda_negedges = list(set(self.gda_negedges))
            self.pos_statement = list(set(self.pos_statement))
            # self.neg_statement = list(set(self.neg_statement))
            self.subclass_edges = list(set(self.subclass_edges))
            self._create_node_mapping()

            cache_data = {'node2id': self.node2id, 'gda_edges': self.gda_edges,
                #'gda_negedges': self.gda_negedges, 
                'pos_statement': self.pos_statement,
                #'neg_statement': self.neg_statement, 
                'subclass_edges': self.subclass_edges}
            torch.save(cache_data, cache_path)

        if graph_type == 'dgl': self.graph = self._build_dgl_graph()
        elif graph_type == 'pyg': self.graph = self._build_pyg_graph()
        else: raise ValueError("graph_type must be 'dgl' or 'pyg'")

        if self.use_pstatement_sampler and 'pos_statement' in self.graph.etypes:
            src, dst = self.graph.edges(etype='pos_statement')
            self.state_list = list(zip(src.tolist(), dst.tolist()))
            eids = torch.arange(self.graph.num_edges('pos_statement'), device=src.device)
            self.graph = dgl.remove_edges(self.graph, eids, etype='pos_statement')

        # elif self.use_nstatement_sampler and 'neg_statement' in self.graph.etypes:
        #     src, dst = self.graph.edges(etype='neg_statement')
        #     self.state_list = list(zip(src.tolist(), dst.tolist()))
        #     eids = torch.arange(self.graph.num_edges('neg_statement'), device=src.device)
        #     self.graph = dgl.remove_edges(self.graph, eids, etype='neg_statement')



    def _read_files(self) -> None:
        """
        Reads data from .tsv or .owl files.
        This expects an instantiated ontology in the owl file; and a tsv with data links. 
        e.g., for a PPI KG, a tsv with PPIs and an .owl with the annotations to the ontology.
        """
        if not hasattr(self, "obsolete_edges"): self.obsolete_edges = []
        pos_st, neg_st = 0, 0
        for fname in os.listdir(self.folder):
            path = os.path.join(self.folder, fname)
            if not os.path.isfile(path): continue

            if fname.lower().endswith('.tsv'):
                df = pd.read_csv(path, sep='\t', header=None)
                subj = (df.iloc[:, 0].str.strip().str.split("/").str[-1])
                obj = (df.iloc[:, 1].str.strip().str.split("/").str[-1])
                lbl  = df.iloc[:, -1].astype(int)
                for s, o, l in zip(subj, obj, lbl):
                    if l == 1: self.gda_edges.append((s, o))
                    # else: self.gda_negedges.append((s, o))

            elif fname.lower().endswith('.owl'):
                g = Graph()
                g.parse(path, format='xml')
                PROPERTY_ASSERTION = URIRef("http://www.w3.org/2002/07/owl#PropertyAssertion")

                def is_class(n):
                    return ((n, RDF.type, OWL.Class) in g or (n, RDF.type, RDFS.Class) in g or
                            any(g.triples((n, RDFS.subClassOf, None))) or any(g.triples((None, RDFS.subClassOf, n))))

                def is_property(n):
                    return ((n, RDF.type, RDF.Property) in g or (n, RDF.type, OWL.ObjectProperty) in g or
                            (n, RDF.type, OWL.DatatypeProperty) in g or (n, RDF.type, OWL.AnnotationProperty) in g)

                def is_individual(n):
                    if isinstance(n, BNode): return False
                    return not ((n, RDF.type, OWL.Class) in g or (n, RDF.type, RDFS.Class) in g or is_property(n))

                @staticmethod
                def local(n):
                    n = str(n)
                    if "#" in n: return n.rsplit("#", 1)[-1]
                    return n.rsplit("/", 1)[-1]

                @staticmethod
                def literal_is_true(lit: Literal) -> bool:
                    if not isinstance(lit, Literal): return False
                    py = lit.toPython()
                    if isinstance(py, bool): return py
                    return str(lit).strip().lower() in ("true", "1")


                DEPRECATED_PROPS = [OWL.deprecated,  # owl:deprecated true
                    URIRef("http://www.geneontology.org/formats/oboInOwl#deprecated")]
                DEPRECATED_CLASSES = [URIRef("http://www.w3.org/2002/07/owl#DeprecatedClass"),
                    URIRef("http://www.geneontology.org/formats/oboInOwl#ObsoleteClass")]

                obsolete_nodes = set()
                for prop in DEPRECATED_PROPS:
                    for s in g.subjects(prop, None):
                        for o in g.objects(s, prop):
                            if literal_is_true(o): obsolete_nodes.add(s)

                for dep_cls in DEPRECATED_CLASSES:
                    for s in g.subjects(RDF.type, dep_cls):
                        obsolete_nodes.add(s)
                obsolete_ids = {local(s) for s in obsolete_nodes}


                def keep_or_mark_obsolete(edge_type, src_id, tgt_id):
                    if (src_id in obsolete_ids) or (tgt_id in obsolete_ids):
                        self.obsolete_edges.append({
                            "file": fname,
                            "edge_type": edge_type,
                            "source": src_id,
                            "target": tgt_id})
                        return False
                    return True


                # PositivePropertyAssertion
                for assertion in g.subjects(RDF.type, PROPERTY_ASSERTION):
                    for src in g.objects(assertion, OWL.sourceIndividual):
                        for tgt in g.objects(assertion, OWL.targetIndividual):
                            s_id, o_id = local(tgt), local(src)
                            if keep_or_mark_obsolete("PositivePropertyAssertion", s_id, o_id):
                                self.pos_statement.append((s_id, o_id))
                                pos_st += 1

                ## IF PROPERTY_ASSERTION GIVES ERROR USE THIS INSTEAD:
                # if OWL.PropertyAssertion in g:
                #     for assertion in g.subjects(RDF.type, OWL.PropertyAssertion):
                #         for src in g.objects(assertion, OWL.sourceIndividual):
                            # for tgt in g.objects(assertion, OWL.targetIndividual):
                            #     s_id, o_id = local(tgt), local(src)
                            #     if keep_or_mark_obsolete("PositivePropertyAssertion", s_id, o_id):
                            #         self.pos_statement.append((s_id, o_id))
                            #         pos_st += 1


                # NegativePropertyAssertion
                # for assertion in g.subjects(RDF.type, OWL.NegativePropertyAssertion):
                # #     for src in g.objects(assertion, OWL.sourceIndividual):
                #         for tgt in g.objects(assertion, OWL.targetIndividual):
                #             s_id, o_id = local(tgt), local(src)
                #             if keep_or_mark_obsolete("NegativePropertyAssertion", s_id, o_id):
                #                 self.pos_statement.append((s_id, o_id))
                #                 neg_st += 1


                # Class Assertion (instance annotation)
                for ind, cls in g.subject_objects(RDF.type):
                    if cls in {OWL.Class, RDFS.Class, OWL.Restriction, OWL.Ontology}: continue
                    if not is_class(cls): continue
                    if not is_individual(ind): continue
                    ind_id = local(ind)
                    cls_id = local(cls)
                    if keep_or_mark_obsolete("ClassAssertion", ind_id, cls_id): self.annotations.append((ind_id, cls_id))
            

                # subClassOf link
                for s, o in g.subject_objects(RDFS.subClassOf):
                    # self.subclass_edges.append((str(s).split('/')[-1], str(o).split('/')[-1]))
                    s_id,o_id = local(s), local(o)
                    if keep_or_mark_obsolete("subClassOf", s_id, o_id):
                        self.subclass_edges.append((s_id, o_id))

                if self.obsolete_edges: self._obsolete_to_csv()


    def _obsolete_to_csv(self) -> None: # saves all relations where a class (subj or obj) is deprecated
        obs_df = pd.DataFrame(self.obsolete_edges, sep="\t").drop_duplicates()
        out_path = os.path.join(self.folder, "obsolete_classes.tsv")
        obs_df.to_csv(out_path, index=False)
        print(f"Saved obsolete relations to: {out_path}")


    def _create_node_mapping(self) -> None:
        """
        Maps root entity IDs to node Idxs.
        """
        nodes = set()
        for edge_list in [self.gda_edges, self.pos_statement, # self.gda_negedges, self.neg_statement, 
                          self.subclass_edges]:
            for s, t in edge_list:
                nodes.add(s)
                nodes.add(t)
        self.node2id = {uri: idx for idx, uri in enumerate(sorted(nodes))}


    def _build_dgl_graph(self) -> dgl.DGLHeteroGraph:
        """
        For building a DGL HeteroGraph.
        """
        def to_tensor(pairs):
            src = [self.node2id[s] for s, _ in pairs]
            dst = [self.node2id[t] for _, t in pairs]
            return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)

        data_dict = {}
        if self.gda_edges:
            data_dict[('node', 'gda', 'node')] = to_tensor(self.gda_edges)
        if self.pos_statement:
            data_dict[('node', 'pos_statement', 'node')] = to_tensor(self.pos_statement)
        # if self.neg_statement:
        #     data_dict[('node', 'neg_statement', 'node')] = to_tensor(self.neg_statement)
        if self.subclass_edges:
            data_dict[('node', 'link', 'node')] = to_tensor(self.subclass_edges)
        
        g = dgl.heterograph(data_dict)
        num_nodes = g.num_nodes('node')
        g.nodes['node'].data['feat'] = torch.randn(num_nodes, self.feature_dim)
        return g
    

    def _build_pyg_graph(self) -> torch_geometric.data.HeteroData:
        """
        For building a PyG HeteroGraph.
        """
        data = HeteroData()
        num_nodes = len(self.node2id)
        data['node'].num_nodes = num_nodes
        feat = torch.randn(num_nodes, self.feature_dim)
        data['node'].feat = feat

        def add_edges(rel_name: str, pairs: list[tuple[str, str]]):
            if not pairs: return
            src_ids = [self.node2id[s] for s, _ in pairs]
            dst_ids = [self.node2id[t] for _, t in pairs]
            edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
            data[('node', rel_name, 'node')].edge_index = edge_index

        add_edges('gda', self.gda_edges)
        add_edges('pos_statement', self.pos_statement)
        # add_edges('neg_statement', self.neg_statement)
        add_edges('link', self.subclass_edges)
        return data


    def get_graph(self) -> Union[dgl.DGLHeteroGraph, torch_geometric.data.HeteroData]:
        """
        Returns the graph variable.
        """
        return self.graph

    def get_uri_mapping(self) -> dict:
        """
        Returns a node's root URI.
        """
        return self.node2id


    def get_edge_lists(self) -> dict:
        return {
            'gda': [(self.node2id[s], self.node2id[t]) for s, t in self.gda_edges],
            'pos_statement': [(self.node2id[s], self.node2id[t]) for s, t in self.pos_statement],
            # 'neg_statement': [(self.node2id[s], self.node2id[t]) for s, t in self.neg_statement],
            'link': [(self.node2id[s], self.node2id[t]) for s, t in self.subclass_edges],
        }


    def get_nodes_with_edge_type(self, etype: str) -> list:
        """
        Return all node IDs that participate (as source or target) in edges of the specified type.
        """
        if etype not in self.graph.etypes:
            raise ValueError(f"Edge type '{etype}' not in graph. Available types: {self.graph.etypes}")

        src, dst = self.graph.edges(etype=etype)
        unique_nodes = set(src.tolist()) | set(dst.tolist())
        return sorted(unique_nodes)


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