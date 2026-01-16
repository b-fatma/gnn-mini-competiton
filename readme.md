# 🎬 MovieLens Rating Prediction Challenge

Graph Neural Networks for Bipartite Recommendation Systems with Cold Start

---

## 🎯 Challenge Overview

Welcome to the **MovieLens Rating Prediction Challenge!** This competition focuses on building Graph Neural Networks to predict movie ratings in a bipartite user-movie network.

**Primary Goal:** Build a GNN model to predict user-movie ratings, handling the unique challenges of bipartite graphs and cold start users/movies.

📊 **View Live Leaderboard:** [LEADERBOARD.md](LEADERBOARD.md)

---

## 📋 Problem Description

### The Task
Given a bipartite graph of users and movies with historical ratings, predict ratings for unseen user-movie pairs.

**Input:**
- User-movie bipartite graph with rating edges
- User features (age, gender, occupation)
- Movie features (19 binary genre indicators)
- Training ratings: ~80,000 user-movie interactions

**Output:**
- Predicted ratings for test set (1.0-5.0 scale)
- Test set contains both warm-start and **cold-start** scenarios

### Key Challenges 🎯

#### 1. **Bipartite Graph Structure** (Primary Challenge)
The fundamental issue: users and movies are **two different node types** in a bipartite graph.

**Why This Matters for Message Passing:**
```
Traditional GNNs:      User-User GNN:        Movie-Movie GNN:
  All nodes same type     But we need both!      But we need both!

BIPARTITE CHALLENGE:
  Users ←→ Movies
   (type 0)  (type 1)
  
Message passing must:
  ✗ NOT mix user and movie embeddings directly
  ✓ Use separate aggregation for each node type
  ✓ Handle different feature dimensions
  ✓ Preserve node type information through layers
  
Tricky Part: How to combine user info + movie info for prediction?
```

**Solutions to Explore:**
- Separate embedding spaces for users and movies
- Bipartite-specific GNN architectures
- Attention mechanisms to weight user-movie interactions
- Distinct encoder networks per node type

#### 2. **Cold Start Problem**
~25-30% of test set contains cold-start users or movies (≤5 interactions in training).

**Why It's Hard:**
- Cold start users/movies have very limited interaction history
- GNN aggregation has few neighbors to learn from
- Cannot rely on rich neighborhood information
- Model must generalize from very sparse data

**What Makes It Cold:**
```
Warm Start User:        Cold Start User:
├─ 50+ ratings           ├─ 1-5 ratings
├─ Rich neighborhood     ├─ Few neighbors
└─ Good embeddings       └─ Weak embeddings
   EASY to predict          HARD to predict
```

---

## 📊 Dataset

**Source:** MovieLens 100K Dataset (ml-100k)

### Structure

```
Users:   943 users
Movies:  1,682 movies
Edges:   100,000 ratings
Time:    Historical data
```

### Files in `data/`

| File | Description |
|------|-------------|
| `graph_data.pt` | Bipartite graph (PyTorch): node features, edge index, edge attributes |
| `train_ratings.csv` | Training ratings: user_id, movie_id, rating |
| `test_pairs.csv` | Test user-movie pairs (no labels) |
| `test_ratings_hidden.csv` | Ground truth for evaluation (hidden) |
| `metadata.pkl` | Dataset statistics and encoders |

### Dataset Characteristics

```python
{
    'num_users': 943,
    'num_movies': 1682,
    'num_train_ratings': 80114,
    'num_test_ratings': 20774,
    'rating_range': [1.0, 5.0],
    'sparsity': 6.3%,  # Very sparse!
    'cold_start_threshold': 5,  # Users/movies with ≤5 ratings
    'cold_start_fraction': 0.27  # ~27% of test set
}
```

### Node Features

**User Features (24-dim):**
- Age (normalized to 0-1)
- Gender (binary: M/F)
- Occupation (one-hot encoded: 21 categories)

**Movie Features (19-dim):**
- Genres (one-hot encoded: Action, Comedy, Drama, Horror, Romance, etc.)

---

## 🎯 Evaluation Metric

**Primary Metric:** RMSE (Root Mean Squared Error)

Lower is better!

```
RMSE = √(mean((y_true - y_pred)²))
```

**Additional Metrics Reported:**
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)

---

## 🤝 How to Submit

### Step 1: Set Up
```bash
git clone https://github.com/YOUR-USERNAME/gnn
cd gnn
```

### Step 2: Prepare Your Submission

Create predictions for test set:
```python
# Your model code...
predictions = model.predict(test_pairs)

# Save as CSV
submission = pd.DataFrame({
    'user_id': test_pairs['user_id'],
    'movie_id': test_pairs['movie_id'],
    'rating': predictions  # Must be in range [1.0, 5.0]
})
submission.to_csv('submissions/your_name_submission.csv', index=False)
```

### Step 3: Test Locally
```bash
# Test scoring script
cd starter_code
python3 scoring_script.py ../submissions/your_name_submission.csv
```

### Step 4: Submit via GitHub
```bash
# Create submission branch
git checkout -b submission/your_name

# Add your submission
git add submissions/your_name_submission.csv
git commit -m "Submission: your model description"
git push origin submission/your_name
```

**Create a Pull Request on GitHub:**
1. Go to: https://github.com/b-fatma/gnn-mini-competition
2. Click "New Pull Request"
3. GitHub Actions will automatically score your submission
4. Check PR comments for results
5. Your score appears on leaderboard

---

## 🏆 Leaderboard

**Automatically updated** after each submission!

View here: [LEADERBOARD.md](LEADERBOARD.md)

Shows:
- **Rank**: Your position
- **Model**: Submission name
- **RMSE**: Primary metric (lower is better)
- **MAE**: Mean absolute error
- **MAPE**: Percentage error

---

## 📚 Getting Started

### 1. Explore the Data
```bash
cd starter_code
python3 dataloader.py  # Check dataset statistics
```

### 2. Try the Baseline
```bash
# Random Forest baseline
python3 rf_baseline.py

# Check score
python3 scoring_script.py ../submissions/rf_baseline_submission.csv
```

### 3. Build Your Model

**Starter architectures in `models.py`:**
- `SimpleGNN` - Basic 2-layer GNN
- `GCN` - Graph Convolutional Network
- `GAT` - Graph Attention Network
- `MatrixFactorizationGNN` - Hybrid approach

**Training script:**
```bash
python3 training.py --model SimpleGNN --epochs 100
```

---

## 💡 Key Insights & Tips

### For Handling Bipartite Graphs

1. **Preserve Node Types**: Keep user and movie embeddings separate until prediction
2. **Heterogeneous Aggregation**: Different aggregation functions for different node types
3. **Type-Aware Message Passing**: Only aggregate messages from the other node type
4. **Separate Feature Processing**: Process user features and movie features independently first

### For Cold Start Problem

1. **Strong Feature Engineering**: Rely on node features (age, genres) for cold-start nodes
2. **Hybrid Approaches**: Combine collaborative filtering with content-based features
3. **Transfer Learning**: Warm-start embeddings help initialize cold-start nodes
4. **Attention Mechanisms**: Learn to weight important neighbors (even if few)

### Architecture Recommendations

**Option A: Bipartite GCN**
```
User Features → GCN Layer → User Embeddings
Movie Features → GCN Layer → Movie Embeddings
        ↓
  Bipartite Message Passing
        ↓
  Prediction Head: concat(user_emb, movie_emb) → rating
```

**Option B: Heterogeneous GNN**
- Separate networks for user and movie pathways
- Attention fusion at prediction layer
- Better for handling feature asymmetry

**Option C: Matrix Factorization + GNN Hybrid**
- Baseline matrix factorization
- Enhance with graph structure information
- Good for cold start (falls back to features)

---

## 📁 Repository Structure

```
├── data/
│   ├── graph_data.pt           # Bipartite graph
│   ├── train_ratings.csv       # Training labels
│   ├── test_pairs.csv          # Test set
│   ├── test_ratings_hidden.csv # Ground truth (hidden)
│   └── metadata.pkl            # Dataset info
│
├── starter_code/
│   ├── dataloader.py           # Data loading
│   ├── models.py               # GNN architectures
│   ├── training.py             # Training pipeline
│   ├── rf_baseline.py          # Simple baseline
│   ├── scoring_script.py       # Scoring (used by GitHub Actions)
│   ├── update_leaderboard.py   # Leaderboard updater
│   └── requirements.txt        # Dependencies
│
├── submissions/                # Your submission CSVs
│   └── your_name_submission.csv
│
├── .github/workflows/
│   └── score-submission.yml    # GitHub Actions automation
│
├── LEADERBOARD.md              # Live leaderboard
├── main.ipynb                  # EDA notebook
└── README.md                   # This file
```

---

## 🛠️ Technical Requirements

### Environment
- **Python:** 3.10+
- **PyTorch:** 2.0+
- **PyTorch Geometric:** For GNN layers

### Installation
```bash
pip install -r starter_code/requirements.txt
```

### Key Dependencies
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
torch==2.0.1
torch-geometric==2.3.1
matplotlib==3.7.2
```

---

## 📚 References & Resources

### Learning Materials
- **GNN Basics:** [PyTorch Geometric Tutorials](https://pytorch-geometric.readthedocs.io/)
- **Bipartite Graphs:** [Bipartite GNN Paper](https://arxiv.org/abs/2010.12013)
- **Cold Start:** [Exploration vs Exploitation in Recommenders](https://arxiv.org/abs/2010.08268)
- **Graph Attention:** [Graph Attention Networks](https://arxiv.org/abs/1710.10903)

### Datasets
- **MovieLens 100K:** [Source](https://grouplens.org/datasets/movielens/100k/)
- **MovieLens Full:** [Source](https://grouplens.org/datasets/movielens/)

### Recommender Systems
- Netflix Prize winners' reports
- Collaborative filtering surveys
- Knowledge Graph Embedding methods

---

## 📝 Important Notes

### What You CAN Use
✅ Provided node features (user age/gender/occupation, movie genres)  
✅ Graph structure (user-movie edges and ratings)  
✅ Any standard GNN architecture (GCN, GAT, GraphSAGE, etc.)  
✅ Ensemble methods  
✅ Pre-trained embeddings from the dataset itself  

### What You CANNOT Use
❌ External user/movie data  
❌ IMDB metadata beyond genres  
❌ Pre-trained models from other datasets  
❌ Web scraping or external APIs  

---

## 🏅 Citation

If you use this challenge in research, please cite:

```
@misc{MovieLensGNNChallenge2026,
  title={MovieLens Rating Prediction: GNN Challenge for Bipartite Graphs with Cold Start},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/YOUR-USERNAME/gnn}},
  note={Bipartite GNN Competition}
}
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## ❓ Questions?

- Check existing issues on GitHub
- Read the starter code comments
- Refer to dataset documentation in `metadata.pkl`
- Review baseline implementations in `rf_baseline.py` and `training.py`

---

**Good luck and happy modeling! 🚀**

May your GNN handle bipartite graphs well and your cold start embeddings be strong! ❄️→🔥
