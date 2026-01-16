# 🎬 MovieLens GNN Rating Prediction Challenge

Build Graph Neural Networks to predict movie ratings in a bipartite user-movie network.

👉 **[View Live Leaderboard](LEADERBOARD.md)**

---

## 🎯 Challenge

Predict movie ratings (1-5 stars) for user-movie pairs using GNNs.

**Dataset:** MovieLens 100K
- 943 users, 1,682 movies
- 80,000 training ratings, 20,000 test pairs
- User features: age, gender, occupation
- Movie features: genres

**Goal:** Achieve RMSE < 0.95 (baseline: ~1.15)

---

## What's Challenging?

### 1. Bipartite Graph Structure
Users and movies are **different node types** - traditional GNNs struggle with this:
```
Users ←→ Movies
(type 0)  (type 1)

✗ Can't mix embeddings directly
✓ Need type-aware architectures
✓ Separate aggregation per type
```

### 2. Cold Start Problem
~27% of test pairs involve users/movies with ≤5 training ratings.

### 3. Extreme Sparsity
Only 6.3% of user-movie pairs have ratings.

---

## 🚀 Quick Start

```bash
# Clone repo
git clone https://github.com/YOUR-USERNAME/movielens-gnn-challenge.git
cd movielens-gnn-challenge

# Install dependencies
pip install -r starter_code/requirements.txt

# Run Random Forest baseline (~1.15 RMSE)
cd starter_code
python rf_baseline.py
# This generates: submissions/rf_baseline_submission.csv
```

**Now beat the baseline with GNNs!**

---

## 📊 Evaluation

**Primary Metric:** RMSE (Root Mean Squared Error) - lower is better

```python
RMSE = √(mean((y_true - y_pred)²))
```

---

## 🤝 Submission

### Format
CSV with 3 columns: `user_id`, `movie_id`, `rating` (1.0-5.0)

```csv
user_id,movie_id,rating
1,1,4.2
1,50,3.8
```

### Process

1. **Generate predictions:**
```python
submission = pd.DataFrame({
    'user_id': test_pairs['user_id'],
    'movie_id': test_pairs['movie_id'],
    'rating': predictions
})
submission.to_csv('submissions/your_name.csv', index=False)
```

2. **Submit via PR:**
```bash
git checkout -b submission/your-name
git add submissions/your_name.csv
git commit -m "Submission: your model description"
git push origin submission/your-name
```

4. GitHub Actions auto-scores → leaderboard updates

---

## 💡 Ideas to Try

**Beginner:**
- Tune hyperparameters (lr, hidden_dim, dropout)
- Try GCN, GAT, GraphSAGE
- Add more GNN layers

**Intermediate:**
- Matrix Factorization + GNN hybrid
- Attention mechanisms
- Feature engineering

**Advanced:**
- Heterogeneous GNNs for bipartite graphs
- Graph sampling techniques
- Ensemble models
- Custom loss functions

---

## 📁 Repository

```
├── data/
│   ├── graph_data.pt              # Graph structure
│   ├── train_ratings.csv          # Training labels
│   ├── test_pairs.csv             # Predict these!
│   └── test_ratings_hidden.csv    # Ground truth (hidden)
├── starter_code/
│   ├── dataloader.py              # Data loading
│   ├── models.py                  # GNN models
│   ├── rf_baseline.py             # Baseline
│   └── scoring_script.py          # Evaluation
└── submissions/                   # Your submissions
```

---

## 📋 Rules

**✅ Allowed:**
- Any GNN Architecture
- Hyperparameter tuning
- Feature engineering from provided data

**❌ Not Allowed:**
- External datasets
- Pre-trained models
- Using test labels

---

## 🎓 Resources

- **Lectures:** [Basira Lab DGL Course](https://www.youtube.com/channel/UCxsqJMTD-yOe277vtQIRjgw)
- **Docs:** [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- **Dataset:** [MovieLens Info](https://grouplens.org/datasets/movielens/)

---

## 📄 License

MIT License

---

**Good luck! 🚀** Fork, build, and submit to climb the leaderboard!