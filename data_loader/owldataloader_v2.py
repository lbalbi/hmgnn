import os, torch
import pandas as pd
from typing import Union, List, Tuple, Dict
from rdflib import Graph, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL
from torch_geometric.data import HeteroData


class OwlDataLoader:
    """
    DataLoader that reads .tsv and .owl files from a folder,
    extracts gene_disease_associations, ontology assertions, and subclass edges,
    caches intermediate data to a .pt file, and builds a PyG HeteroData graph with numeric node IDs.

    If use_pstatement_sampler or use_nstatement_sampler is True, the corresponding
    statement edges are extracted into `state_list` and removed from the graph.
    """
    CACHE_NAME = 'data_cache.pt'

    def __init__(self,folder: str,use_pstatement_sampler: bool = False,
                 use_nstatement_sampler: bool = False,feature_dim: int = 128):
                
        self.folder = folder
        self.use_pstatement_sampler = use_pstatement_sampler
        self.use_nstatement_sampler = use_nstatement_sampler
        self.feature_dim = feature_dim
        self.state_list: List[Tuple[int, int]] = []
        cache_path = os.path.join(self.folder, self.CACHE_NAME)

        if os.path.exists(cache_path):
            cache = torch.load(cache_path)
            self.node2id = cache['node2id']
            self.gda_edges = cache['gda_edges']
            self.gda_negedges = cache.get('gda_negedges', [])
            self.pos_statement = cache['pos_statement']
            self.neg_statement = cache.get('neg_statement', [])
            self.subclass_edges = cache['subclass_edges']
        else:
            self.gda_edges: List[Tuple[str, str]] = []
            self.gda_negedges: List[Tuple[str, str]] = []
            self.pos_statement: List[Tuple[str, str]] = []
            self.neg_statement: List[Tuple[str, str]] = []
            self.subclass_edges: List[Tuple[str, str]] = []
            self._read_files()
            self.gda_edges = list(set(self.gda_edges))
            self.gda_negedges = list(set(self.gda_negedges))
            self.pos_statement = list(set(self.pos_statement))
            self.neg_statement = list(set(self.neg_statement))
            self.subclass_edges = list(set(self.subclass_edges))
            self._create_node_mapping()

            cache_data = {'node2id': self.node2id,
                'gda_edges': self.gda_edges,
                'gda_negedges': self.gda_negedges,
                'pos_statement': self.pos_statement,
                'neg_statement': self.neg_statement,
                'subclass_edges': self.subclass_edges}
            torch.save(cache_data, cache_path)

        self.graph: HeteroData = self._build_pyg_graph()
        if self.use_pstatement_sampler: self._extract_and_remove_edges('pos_statement')
        elif self.use_nstatement_sampler: self._extract_and_remove_edges('neg_statement')


    def _read_files(self) -> None:
        """
        Reads data from .tsv or .owl files.
        """
        if not hasattr(self, "obsolete_edges"): self.obsolete_edges = []
        self.annotations: List[Tuple[str, str]] = []

        for fname in os.listdir(self.folder):
            path = os.path.join(self.folder, fname)
            if not os.path.isfile(path): continue
            if fname.lower().endswith('.tsv'):
                df = pd.read_csv(path, sep='\t', header=None)
                subj = (df.iloc[:, 0].str.strip().str.split("/").str[-1])
                obj = (df.iloc[:, 1].str.strip().str.split("/").str[-1])
                lbl = df.iloc[:, -1].astype(int)
                for s, o, l in zip(subj, obj, lbl):
                    if l == 1: self.gda_edges.append((s, o))
                    else: self.gda_negedges.append((s, o))
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

                def local(n):
                    n = str(n)
                    if "#" in n: return n.rsplit("#", 1)[-1]
                    return n.rsplit("/", 1)[-1]

                def literal_is_true(lit: Literal) -> bool:
                    if not isinstance(lit, Literal): return False
                    py = lit.toPython()
                    if isinstance(py, bool): return py
                    return str(lit).strip().lower() in ("true", "1")

                DEPRECATED_PROPS = [
                    OWL.deprecated,  # owl:deprecated true
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
                        self.obsolete_edges.append({"file": fname,"edge_type": edge_type,
                            "source": src_id,"target": tgt_id})
                        return False
                    return True

                # PositivePropertyAssertion
                for assertion in g.subjects(RDF.type, PROPERTY_ASSERTION):
                    for src in g.objects(assertion, OWL.sourceIndividual):
                        for tgt in g.objects(assertion, OWL.targetIndividual):
                            s_id, o_id = local(tgt), local(src)
                            if keep_or_mark_obsolete("PositivePropertyAssertion", s_id, o_id):
                                self.pos_statement.append((s_id, o_id))

                # NegativePropertyAssertion
                for assertion in g.subjects(RDF.type, OWL.NegativePropertyAssertion):
                    for src in g.objects(assertion, OWL.sourceIndividual):
                        for tgt in g.objects(assertion, OWL.targetIndividual):
                            s_id, o_id = local(tgt), local(src)
                            if keep_or_mark_obsolete("NegativePropertyAssertion", s_id, o_id):
                                self.neg_statement.append((s_id, o_id))

                # Class Assertion
                for ind, cls in g.subject_objects(RDF.type):
                    if cls in {OWL.Class, RDFS.Class, OWL.Restriction, OWL.Ontology}: continue
                    if not is_class(cls): continue
                    if not is_individual(ind): continue
                    ind_id = local(ind)
                    cls_id = local(cls)
                    if keep_or_mark_obsolete("ClassAssertion", ind_id, cls_id):
                        self.annotations.append((ind_id, cls_id))

                # subClassOf links
                for s, o in g.subject_objects(RDFS.subClassOf):
                    s_id, o_id = local(s), local(o)
                    if keep_or_mark_obsolete("subClassOf", s_id, o_id):
                        self.subclass_edges.append((s_id, o_id))
                if self.obsolete_edges: self._obsolete_to_csv()

    def _obsolete_to_csv(self) -> None:
        obs_df = pd.DataFrame(self.obsolete_edges).drop_duplicates()
        out_path = os.path.join(self.folder, "obsolete_classes.tsv")
        obs_df.to_csv(out_path, index=False, sep="\t")
        print(f"Saved obsolete relations to: {out_path}")

    def _create_node_mapping(self) -> None:
        """
        Maps root entity IDs to node Idxs.
        """
        nodes = set()
        for edge_list in [self.gda_edges, self.pos_statement, self.neg_statement, self.subclass_edges]:
            for s, t in edge_list:
                nodes.add(s)
                nodes.add(t)
        self.node2id: Dict[str, int] = {uri: idx for idx, uri in enumerate(sorted(nodes))}


    def _build_pyg_graph(self) -> HeteroData:
        """
        Build a PyTorch Geometric HeteroData graph.
        """
        data = HeteroData()
        num_nodes = len(self.node2id)
        data['node'].num_nodes = num_nodes
        data['node'].feat = torch.randn(num_nodes, self.feature_dim)

        def add_edges(rel_name: str, pairs: List[Tuple[str, str]]):
            if not pairs: return
            src_ids = [self.node2id[s] for s, _ in pairs]
            dst_ids = [self.node2id[t] for _, t in pairs]
            edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
            data[('node', rel_name, 'node')].edge_index = edge_index
        add_edges('gda', self.gda_edges)
        add_edges('pos_statement', self.pos_statement)
        add_edges('neg_statement', self.neg_statement)
        add_edges('link', self.subclass_edges)
        return data

    
    def _rel_to_etype(self, rel: str) -> Tuple[str, str, str]:
        """Map a relation name to its edge-type tuple ('node', rel, 'node')."""
        for et in self.graph.edge_types:
            if et[1] == rel: return et
        raise ValueError(f"Edge type '{rel}' not in graph. "
                         f"Available: {[et[1] for et in self.graph.edge_types]}")

    def _extract_and_remove_edges(self, rel: str) -> None:
        """
        For the given relation, copy its edges into state_list and then remove them from the graph.
        """
        et = self._rel_to_etype(rel)
        if 'edge_index' not in self.graph[et]: return
        ei = self.graph[et].edge_index
        self.state_list = list(zip(ei[0].tolist(), ei[1].tolist()))
        self.graph[et].edge_index = torch.empty((2, 0), dtype=torch.long)


    def get_graph(self) -> HeteroData:
        """Returns the graph variable."""
        return self.graph

    def get_uri_mapping(self) -> dict:
        """Returns a node's root URI -> numeric ID mapping."""
        return self.node2id

    def get_edge_lists(self) -> dict:
        return {'gda': [(self.node2id[s], self.node2id[t]) for s, t in self.gda_edges],
            'pos_statement': [(self.node2id[s], self.node2id[t]) for s, t in self.pos_statement],
            'neg_statement': [(self.node2id[s], self.node2id[t]) for s, t in self.neg_statement],
            'link': [(self.node2id[s], self.node2id[t]) for s, t in self.subclass_edges]}

    def get_nodes_with_edge_type(self, etype: str) -> list:
        """
        Return all node IDs that participate (as source or target) in edges of the specified type.
        """
        et = self._rel_to_etype(etype)
        if 'edge_index' not in self.graph[et]: return []
        src, dst = self.graph[et].edge_index
        return sorted(set(src.tolist()) | set(dst.tolist()))

    def get_negative_edges(self) -> torch.Tensor:
        """
        Returns a 2*E tensor of source and target node IDs for gene-disease negative edges.
        """
        if not getattr(self, 'gda_negedges', None):
            return torch.empty((2, 0), dtype=torch.long)
        src_ids = [self.node2id[s] for s, _ in self.gda_negedges]
        dst_ids = [self.node2id[t] for _, t in self.gda_negedges]
        src = torch.tensor(src_ids, dtype=torch.long)
        dst = torch.tensor(dst_ids, dtype=torch.long)
        return torch.stack([src, dst], dim=0)

    def get_state_list(self) -> list:
        """
        Returns the extracted statement edge list (pos_statement or neg_statement) if a sampler was used.
        """
        return self.state_list
