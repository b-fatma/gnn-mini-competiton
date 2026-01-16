"""
📊 RANDOM FOREST BASELINE - MovieLens Rating Prediction Challenge

This is a SIMPLE baseline to get you started.
Goal: Use basic ML to predict movie ratings, then beat it with a better approach!

How it works:
1. Load training data (user features + movie features + ratings)
2. Train a Random Forest on concatenated features
3. Evaluate on validation set
4. Make predictions on test set
5. Submit!
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import os

from dataloader import load_movielens_data


def main():
    """
    Simple Random Forest baseline pipeline
    """
    
    print("=" * 70)
    print("🎬 RANDOM FOREST BASELINE - MovieLens Rating Prediction")
    print("=" * 70)
    
    # ========================
    # 1. LOAD DATA
    # ========================
    print("\n📚 Loading data...")
    data, dataset = load_movielens_data('../data')
    dataset.print_statistics()
    
    # Get training data
    train_user_idx = dataset.train_user_idx.numpy()
    train_movie_idx = dataset.train_movie_idx.numpy()
    train_ratings = dataset.train_ratings_tensor.numpy()
    
    # Extract features for training data
    user_features = data.x[dataset.train_user_idx].numpy()
    movie_features = data.x[dataset.train_movie_idx].numpy()
    
    # Combine features: [user_feature_1, ..., user_feature_n, 
    #                    movie_feature_1, ..., movie_feature_m]
    X = np.concatenate([user_features, movie_features], axis=1)
    y = train_ratings
    
    print(f"\n✅ Feature matrix shape: {X.shape}")
    print(f"   - Users: {dataset.num_users}")
    print(f"   - Movies: {dataset.num_movies}")
    print(f"   - Feature dim: {X.shape[1]}")
    
    # ========================
    # 2. SPLIT TRAIN/VALIDATION
    # ========================
    print("\n📊 Splitting train/validation (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"   Train: {len(X_train)} ratings")
    print(f"   Valid: {len(X_val)} ratings")
    
    # ========================
    # 3. TRAIN RANDOM FOREST
    # ========================
    print("\n🌲 Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200,      # Number of trees
        max_depth=15,          # Tree depth
        min_samples_split=10,  # Min samples to split a node
        min_samples_leaf=5,    # Min samples in leaf
        random_state=42,
        n_jobs=-1              # Use all CPUs
    )
    
    rf.fit(X_train, y_train)
    print("   ✅ Training complete!")
    
    # ========================
    # 4. EVALUATE ON VALIDATION
    # ========================
    print("\n📈 Evaluating on validation set...")
    y_val_pred = rf.predict(X_val)
    
    # Clip predictions to valid rating range [1, 5]
    y_val_pred = np.clip(y_val_pred, 1.0, 5.0)
    
    # Compute RMSE (Root Mean Squared Error)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"   Validation RMSE: {val_rmse:.4f}")
    
    # ========================
    # 5. PREDICT ON TEST SET
    # ========================
    print("\n🎯 Making predictions on test set...")
    test_user_idx = dataset.test_user_idx
    test_movie_idx = dataset.test_movie_idx
    
    test_user_features = data.x[test_user_idx].numpy()
    test_movie_features = data.x[test_movie_idx].numpy()
    
    X_test = np.concatenate([test_user_features, test_movie_features], axis=1)
    test_predictions = rf.predict(X_test)
    
    # Clip to valid range
    test_predictions = np.clip(test_predictions, 1.0, 5.0)
    
    print(f"   Predicted {len(test_predictions)} test ratings")
    print(f"   Prediction range: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")
    
    # ========================
    # 6. SAVE SUBMISSION
    # ========================
    print("\n💾 Saving submission...")
    
    submission = pd.DataFrame({
        'user_id': dataset.test_pairs['user_id'].values,
        'movie_id': dataset.test_pairs['movie_id'].values,
        'rating': test_predictions
    })
    
    # Create output directory if needed
    os.makedirs('../submissions', exist_ok=True)
    
    output_file = '../submissions/rf_baseline_submission.csv'
    submission.to_csv(output_file, index=False)
    
    print(f"   ✅ Saved to {output_file}")
    print(f"\n   Preview:")
    print(submission.head(10))
    
    # ========================
    # 7. DONE!
    # ========================
    print("\n" + "=" * 70)
    print("🎉 BASELINE COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 Results Summary:")
    print(f"   Validation RMSE: {val_rmse:.4f}")
    print(f"   Test predictions: {len(test_predictions)}")
    print(f"   Submission: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
