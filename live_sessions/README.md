# GNNs Through Three Lenses: Generation, Unification, Cognition

---

## Background: The Two Core Pathologies

Before examining the three lenses, two failure modes recur across all of them.

**Over-smoothing** occurs as GNN depth increases: heterophilic edges mix class-discriminative information across the graph, causing node embeddings to converge to the same value. Formally, `Σ_{(u,v)∈E: Φ(u)≠Φ(v)} ||X^T_u − X^T_v|| → 0` as `T → ∞`. The root cause is not depth per se but the *heterophilic connections* that accelerate convergence. The remedy is to reduce heterophily, not just sparsify.

**Over-squashing** occurs at graph bottlenecks: a node's receptive field grows exponentially with depth, and all that information must compress into a fixed-size vector when passing through a narrow bridge. Nodes near bottlenecks effectively stop receiving meaningful long-range signals. The remedy is to relax or remove bottleneck edges.

These two problems are in **tension**: relaxing bottlenecks (adding edges) often introduces heterophilic connections, worsening over-smoothing. This tension is the central challenge in GNN design.

---

## Lens 1 — Graph Generation

Generative GNNs learn a distribution over graphs and sample from it, or transform one graph into another across domains, resolutions, or time.

### Generative Axes (from Lecture 6)

Three axes describe how source and target graphs can differ:

- **Domain axis** — same system, different modalities (e.g., structural vs. functional brain connectivity from dMRI vs. fMRI). Enables cross-modal synthesis.
- **Resolution axis** — same domain, different granularity (35-node vs. 160-node brain parcellation). Enables graph super-resolution.
- **Time axis** — same system across timepoints. Enables trajectory prediction (e.g., Alzheimer's progression).

### GraphVAE vs. Graph UNet

| | GraphVAE | Graph UNet |
|---|---|---|
| **Goal** | Generate novel graphs from scratch | Learn better embeddings for downstream tasks |
| **Encoder output** | `q_φ(z\|G) = N(μ, σ²)` | Compressed graph via gPool (top-k node selection) |
| **Decoder** | `z ~ N(0,1) → new graph` | gUnpool via stored node indices |
| **Skip connections** | No | Yes (preserve fine-grained structure) |
| **Test-time behaviour** | Sample pure noise → decode | Requires input graph |

### CogGNN — Cognition-Aware Generation

**Problem:** Existing generative GNN models capture structural and topological properties but ignore *cognitive traits* of brain networks.

**Solution:** CogGNN introduces a **Cognitive Reservoir (CR)** — an Echo State Network (ESN) whose internal recurrent weight matrix is initialised from a real brain connectome derived from GNN embeddings. This biologically-grounded design gives the reservoir memory-like dynamics.

**Pipeline:**
1. A generative GNN (specifically DGN — Deep Graph Normalizer) processes multi-view brain networks and produces node embeddings per ROI.
2. The embeddings are converted into a connectome, which seeds the reservoir.
3. The CR maps cognitive input sequences (e.g., MNIST image sequences) into a high-dimensional state space via: `x_i = tanh(α·W_in[1; c_i] + (1−α)·W_res·x_{i-1})`  where `α` controls the memory-current input trade-off.
4. Only the output layer `W_out` is trained, via a **cognitive loss** `L_cog` that minimises reconstruction error between predicted and temporally-delayed images (visual memory recall).

**Joint objective (Vis-CogGNN):**

```
min L = L_centeredness + L_cog
```

The CBT is optimised to be population-representative (centeredness loss) *and* to preserve visual memory (cognitive loss), alternating between the two in a co-optimisation loop. Final CBT: element-wise median over all subject-level templates.

**Results:** Vis-CogGNN outperforms DGN on centeredness in 75% of cases, achieves significantly higher Visual Memory Capacity (p < 0.005 in 6/8 datasets), and produces more discriminative CBTs — up to +22% classification accuracy on AD/LMCI.

---

## Lens 2 — Graph Unification

**Problem:** Different hospitals train heterogeneous models (MLPs, CNNs, GNNs) on local data distributions. Existing paradigms (federated learning, knowledge distillation) assume architectural homogeneity or align output distributions rather than representations.

### uGNN — Unified GNN Learning

**Core idea:** Represent every neural network — regardless of architecture — as a graph, then train a single GNN over the unified graph of all models simultaneously.

**Graph conversion:**

- **MLP → graph:** Neurons are nodes (node feature = bias), weighted connections are directed edges (edge feature = weight). Structure is a DAG.
- **CNN → graph:** Spatial positions at each feature map layer are nodes; receptive field connections are edges with kernel weights.
- **GNN → graph:** Each (node, layer) tuple is a node; edges represent message-passing across layers.

**Unification:** All model-graphs are combined via **disjoint union** — a block-diagonal adjacency structure. No edges cross between models. A shared set of learnable parameters `(θ_edge, θ_bias, θ_edge_shift, θ_bias_shift)` rescales and shifts all edge and node features via a SoftSign nonlinearity, aligning them into a common parametrisation.

**Forward pass:** The uGNN emulates each model's forward pass layer-by-layer using the updated edge and bias features. Final-layer activations are extracted per model and used as predicted logits.

**Training:** Weighted sum of cross-entropy losses across all models: `L = Σ αᵢ Lᵢ`. Gradients update only the shared `θ` parameters — not the individual model parameters directly.

**Why it helps:** Although each model sees only its local distribution during training, the shared parameters facilitate *indirect* knowledge transfer across models and distributions. At test time, models face a mixed distribution they were never exposed to individually.

**Results:** uGNN consistently outperforms individual training on MorphoMNIST, PneumoniaMNIST, and BreastMNIST, with stronger gains under non-IID clustering (engineered distributional shift).

---

## Lens 3 — Graph Cognition / Expressiveness

This lens covers how GNNs reason, what limits their expressiveness, and how to overcome those limits.

### Theoretical Expressiveness (Lecture 5)

GNNs are structurally equivalent to the **Weisfeiler-Leman (WL) test**: both iteratively aggregate neighbourhood information and hash it. This gives an upper bound — a GNN is *at most* as powerful as the WL test.

**Aggregation expressiveness:** Sum > Mean > Max. Sum preserves full multiset structure; mean loses node counts; max loses multiplicities entirely.

**GIN** achieves WL-equivalent expressiveness by replacing the single linear layer with an MLP: `h^{k+1}(v) = MLP((1 + ε) · h^k(v) + Σ_{u∈N(v)} h^k(u))`. The MLP provides universal approximation → injective combination over multisets → no collapsing of distinct neighbourhoods. For graph-level tasks, GIN concatenates embeddings across all K layers before readout.

---

## Addressing Over-Smoothing and Over-Squashing: DuoGNN and DeltaGNN

### DuoGNN — Topology-Aware Interaction Decoupling

**Core idea:** Separate homophilic and heterophilic interactions *before* aggregation, then process them independently.

**Pipeline:**

1. **Topological edge filtering:** Remove the `κ` least connected edges using a connectivity measure (degree, curvature, etc.). This breaks the graph into homophilic connected components `G_ho`. Short-range interactions are preserved; bottlenecks and heterophilic edges are removed.
2. **Heterophilic graph condensation:** Select the most connected node from each of the `μ` most populated clusters of `G_ho`. Connect them all-to-all to form `G_he` — a small, fully-connected, strongly heterophilic graph that preserves long-range interactions (LRIs).
3. **Parallel transformation:** `G_ho` → homophilic GNN module. `G_he` → heterophilic GNN module (no aggregation in first layer; jumping knowledge concatenation across layers). Both outputs concatenated for final prediction.

**Key distinction from rewiring:** Standard rewiring adds edges to relax bottlenecks (worsening homophily) and removes edges to slow smoothing (not fixing its cause). DuoGNN instead *routes* different interactions to different modules, directly targeting the cause of each pathology.

### DeltaGNN — Information Flow Control

**Core idea:** Use embedding dynamics during message passing to identify both bottlenecks (over-squashing) and heterophilic nodes (over-smoothing) simultaneously, via a single novel score.

**Delta embeddings:**
- **First delta** `Δ^t_u = d(⊕_{v∈N(u)} ψ(X^t_v), ψ(X^t_u))` — velocity of aggregation at layer t.
- **Second delta** `(Δ²)^t_u = d(Δ^t_u, Δ^{t-1}_u)` — acceleration of aggregation.

**Theoretical grounding:**
- **Lemma 1:** High mean `Δ_u` → low homophilic ratio `H_u` (detects heterophilic nodes).
- **Lemma 2:** Low variance `V_t[Δ²_u]` → node near a bottleneck (slower, more constrained convergence).

**Information Flow Score (IFS):**

```
S_u = (m · V_t[Δ²_u] + 1) / (l · Δ_u + 1)
```

Minimised for heterophilic bottleneck nodes. Computed *during* message passing with `O(|V|)` complexity — the lowest among all major connectivity measures.

**Information Flow Control (IFC):** Sequential edge-filtering layers interwoven with GNN layers. At each layer, remove edges adjacent to nodes with lowest IFS scores. The filtering threshold `θ` is tuned via hill ascent on the mean node score.

**DeltaGNN architecture:** Extends DuoGNN by replacing static topological filtering with IFC. Same dual pipeline (homophilic aggregation with IFC → heterophilic condensation → heterophilic aggregation → concatenate), but the edge filtering is now embedding-aware and adaptive.

**Results:** Outperforms state-of-the-art on 4/6 homophily-varying datasets (+1.23% avg), 30.61% reduction in epoch time vs. worst baseline, no OOT errors on large graphs (unlike BC, CC, RC), and memory ~2× GCN (vs. GAT which OOMs on dense datasets).

---

## Comparative Summary

| Paper | Lens | Core Problem | Key Mechanism | Over-smoothing | Over-squashing |
|---|---|---|---|---|---|
| **CogGNN** | Generation | Brain templates lack cognitive grounding | Echo State Network reservoir + cognitive loss | ✗ (not addressed) | ✗ |
| **uGNN** | Unification | Heterogeneous models fail under distribution shift | Convert all DNNs to graphs; shared GNN meta-parameters | ✗ | ✗ |
| **DuoGNN** | Cognition/Expressiveness | LRIs lost due to local aggregation paradigm | Topological interaction-decoupling; dual aggregation | ✓ (remove heterophilic edges) | ✓ (remove bottlenecks) |
| **DeltaGNN** | Cognition/Expressiveness | Same, + static measures fail on large/diverse graphs | Embedding-aware IFS; IFC during message passing | ✓ (embedding-guided) | ✓ (embedding-guided) |