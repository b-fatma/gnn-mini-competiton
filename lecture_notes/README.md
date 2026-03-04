# Deep Graph Learning — Lecture Summaries
> Notes based on the **BASIRA Lab DGL Lecture Series**  
> https://basira-lab.com/ | https://github.com/basiralab/DGL

---

## Contents

| File | Lecture | Title | Key Topics |
|---|---|---|---|
| `L1.md` | Lecture 1 | **Classical Graph Representations** | Graph types (directed, simple, complete, hypergraph, heterogeneous), matrix representations (A, X, E), 8 graph/matrix properties (connectivity, walks, spectra, topology, efficiency), the three core learning tasks (graph/node/edge) |
| `L2.md` | Lecture 2 | **Feature Embedding: From ML to GCNs** | Individualistic vs. collectivist learning paradigm, manifold learning, shallow node embeddings, the vanilla GCN propagation rule `H_{k+1} = a(ÂHΩ)`, layer-wise normalization, graph/node/edge feature extraction |
| `L3.md` | Lecture 3 | **GCN Training & Aggregation** | 5-step GCN design process, loss functions, prediction heads (node/edge/graph), inductive testing on unseen nodes, graph-level pooling (mean/max/sum, DiffPool), aggregation strategies (mean, Kipf, max, attention/GAT) |
| `L4.md` | Lecture 4 | **Batching, Sampling & Learning Paradigms** | Graph expansion problem, node sampling (GraphSAGE), layer sampling (FastGCN, LADIES), subgraph sampling (ClusterGCN), batch normalization, dropout & DropGNN, inductive vs. transductive learning |
| `L5.md` | Lecture 5 | **GNN Properties: Invariance, Equivariance & Expressiveness** | Node permutation invariance/equivariance, computational graphs as rooted subtrees, WL graph isomorphism test & color refinement, expressiveness limits of mean/max aggregation, GIN (sum + MLP = WL-equivalent) |
| `L6.md` | Lecture 6 | **Generative GNNs** | Supervised generative tasks (adjacency/node/edge feature generation, loss formulations), unsupervised generation (density estimation + sampling, GraphVAE), generation taxonomy (Guo et al. 2023), Graph UNet (gPool/gUnpool), evaluation metrics (MAE, KLD, MMD, FID) |
