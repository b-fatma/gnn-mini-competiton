# starter_code/baseline.py
"""
BASELINE MODEL - MovieLens Rating Prediction Challenge

This is a simple baseline to get you started.
Your goal: Beat this baseline by improving the model, features, or training!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from dataloader import load_movielens_data
from models import SimpleGNN


def train_baseline(epochs=50, lr=0.01, batch_size=2048):
    """
    Simple baseline training loop
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    data, dataset = load_movielens_data('../data')
    dataset.print_statistics()
    
    # Move to device
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.train_user_idx = data.train_user_idx.to(device)
    data.train_movie_idx = data.train_movie_idx.to(device)
    data.train_ratings = data.train_ratings.to(device)
    
    # Create validation split (10% of training data)
    num_train = len(data.train_user_idx)
    num_val = int(num_train * 0.1)
    indices = torch.randperm(num_train)
    
    val_user = data.train_user_idx[indices[:num_val]]
    val_movie = data.train_movie_idx[indices[:num_val]]
    val_ratings = data.train_ratings[indices[:num_val]]
    
    train_user = data.train_user_idx[indices[num_val:]]
    train_movie = data.train_movie_idx[indices[num_val:]]
    train_ratings = data.train_ratings[indices[num_val:]]
    
    # Initialize simple model
    model = SimpleGNN(
        num_features=data.x.shape[1],
        hidden_dim=64,
        embedding_dim=32,
        dropout=0.5
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"\nTraining for {epochs} epochs...\n")
    
    best_val_rmse = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        # Get embeddings
        embeddings = model(data.x, data.edge_index)
        
        # Train on batches
        num_batches = (len(train_user) + batch_size - 1) // batch_size
        optimizer.zero_grad()
        epoch_loss = 0
        
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, len(train_user))

            u = train_user[start:end]
            m = train_movie[start:end]
            r = train_ratings[start:end]

            pred = model.predict_rating(
                embeddings[u],
                embeddings[m]
            )

            loss = criterion(pred, r)
            loss.backward()
            epoch_loss += loss.item() * len(r)

        optimizer.step()
        
        train_rmse = np.sqrt(epoch_loss / len(train_user))
        
        # Validate
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index)
            user_emb = embeddings[val_user]
            movie_emb = embeddings[val_movie]
            val_pred = model.predict_rating(user_emb, movie_emb)
            val_loss = criterion(val_pred, val_ratings)
            val_rmse = torch.sqrt(val_loss).item()
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            # Save best model
            torch.save(model.state_dict(), 'baseline_best.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")
    
    print(f"\nBest Validation RMSE: {best_val_rmse:.4f}")
    
    return model, dataset, data, device


def generate_submission(model, dataset, data, device, output_file='../submissions/baseline_submission.csv'):
    """
    Generate predictions for test set
    """
    print("\nGenerating test predictions...")
    
    model.eval()
    test_user = data.test_user_idx.to(device)
    test_movie = data.test_movie_idx.to(device)
    
    with torch.no_grad():
        # Get embeddings
        embeddings = model(data.x, data.edge_index)
        
        # Predict in batches
        batch_size = 4096
        predictions = []
        
        for i in tqdm(range(0, len(test_user), batch_size)):
            batch_user = test_user[i:i+batch_size]
            batch_movie = test_movie[i:i+batch_size]
            
            user_emb = embeddings[batch_user]
            movie_emb = embeddings[batch_movie]
            
            pred = model.predict_rating(user_emb, movie_emb)
            predictions.append(pred.cpu())
        
        predictions = torch.cat(predictions).numpy()
    
    # Create submission
    submission = pd.DataFrame({
        'user_id': dataset.test_pairs['user_id'].values,
        'movie_id': dataset.test_pairs['movie_id'].values,
        'rating': predictions
    })
    
    submission.to_csv(output_file, index=False)
    print(f"✅ Submission saved to {output_file}")
    
    return submission


if __name__ == "__main__":
    # Train baseline model
    model, dataset, data, device = train_baseline(epochs=50, lr=0.01)
    
    # Load best model
    model.load_state_dict(torch.load('baseline_best.pt'))
    
    # Generate submission
    generate_submission(model, dataset, data, device)
    
    print("\n" + "="*60)
    print("BASELINE COMPLETE!")
    print("="*60)
    print("\nNow it's your turn to improve on this baseline!")
    print("\nIdeas to try:")
    print("  - Use a different GNN architecture (GCN, GAT)")
    print("  - Tune hyperparameters (hidden_dim, learning rate, dropout)")
    print("  - Add more GNN layers")
    print("  - Try the MatrixFactorizationGNN hybrid model")
    print("  - Engineer better node features")
    print("  - Use different aggregation methods")
    print("\nGood luck!")
    print("="*60)