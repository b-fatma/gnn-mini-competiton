"""
🎯 SCORING SCRIPT - MovieLens Rating Prediction Challenge

This script evaluates submissions against ground truth labels.
Used by organizers to score participant submissions.

Usage:
    python scoring_script.py <path_to_submission_csv>

Example:
    python scoring_script.py ../submissions/submission.csv
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import os


def get_repo_root():
    """
    Find the repository root by looking for data/ directory or .git folder
    """
    current = os.path.abspath(os.path.dirname(__file__))
    while current != os.path.dirname(current):  # Stop at filesystem root
        if os.path.exists(os.path.join(current, 'data')) or os.path.exists(os.path.join(current, '.git')):
            return current
        current = os.path.dirname(current)
    # Fallback to current directory
    return os.getcwd()


def load_files(submission_file):
    """
    Load submission and ground truth files
    """
    # Find repo root
    repo_root = get_repo_root()
    
    # Convert submission file to absolute path if relative
    if not os.path.isabs(submission_file):
        submission_file = os.path.join(repo_root, submission_file)
    
    # Load submission
    if not os.path.exists(submission_file):
        raise FileNotFoundError(f"Submission file not found: {submission_file}")
    
    submission = pd.read_csv(submission_file)
    
    # Load ground truth (hidden test set) - always from repo root
    truth_file = os.path.join(repo_root, 'data', 'test_ratings_hidden.csv')
    if not os.path.exists(truth_file):
        raise FileNotFoundError(f"Ground truth file not found: {truth_file}")
    
    truth = pd.read_csv(truth_file)
    
    return submission, truth


def validate_submission(submission):
    """
    Validate submission format
    """
    required_columns = ['user_id', 'movie_id', 'rating']
    
    # Check columns
    missing_cols = [col for col in required_columns if col not in submission.columns]
    if missing_cols:
        raise ValueError(f"Submission missing columns: {missing_cols}")
    
    # Check for NaN values
    if submission.isnull().any().any():
        raise ValueError("Submission contains NaN values")
    
    # Check rating range
    if (submission['rating'] < 1.0).any() or (submission['rating'] > 5.0).any():
        raise ValueError("Ratings must be in range [1.0, 5.0]")
    
    print("✅ Submission format valid")


def merge_predictions(submission, truth):
    """
    Merge submission with ground truth by user_id and movie_id
    """
    merged = pd.merge(
        submission[['user_id', 'movie_id', 'rating']],
        truth[['user_id', 'movie_id', 'rating']],
        on=['user_id', 'movie_id'],
        suffixes=('_pred', '_true')
    )
    
    if len(merged) == 0:
        raise ValueError("No matching user_id/movie_id pairs between submission and ground truth")
    
    if len(merged) < len(truth):
        missing = len(truth) - len(merged)
        print(f"⚠️  Warning: {missing} ground truth ratings not in submission")
    
    return merged


def compute_metrics(merged):
    """
    Compute evaluation metrics
    """
    y_true = merged['rating_true'].values
    y_pred = merged['rating_pred'].values
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'count': len(merged)
    }


def main():
    """
    Main scoring function
    """
    print("=" * 70)
    print("🎬 MOVIE LENS RATING PREDICTION - SCORING SCRIPT")
    print("=" * 70)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\n❌ Error: Please provide submission file path")
        print("Usage: python scoring_script.py <submission_file>")
        print("Example: python scoring_script.py submissions/submission.csv")
        sys.exit(1)
    
    submission_file = sys.argv[1]
    repo_root = get_repo_root()
    
    try:
        # Load files
        print(f"\n📂 Loading submission: {submission_file}")
        submission, truth = load_files(submission_file)
        
        print(f"   Submission: {len(submission)} predictions")
        print(f"   Ground truth: {len(truth)} ratings")
        
        # Validate submission
        print("\n✔️  Validating submission...")
        validate_submission(submission)
        
        # Merge predictions with ground truth
        print("\n📊 Merging predictions with ground truth...")
        merged = merge_predictions(submission, truth)
        print(f"   Matched: {len(merged)} ratings")
        
        # Compute metrics
        print("\n🎯 Computing metrics...")
        metrics = compute_metrics(merged)
        
        # Print results
        print("\n" + "=" * 70)
        print("📈 RESULTS")
        print("=" * 70)
        print(f"Predictions evaluated: {metrics['count']}")
        print(f"\nRMSE (Root Mean Squared Error):  {metrics['rmse']:.4f}")
        print(f"MAE  (Mean Absolute Error):     {metrics['mae']:.4f}")
        print(f"MAPE (Mean Absolute % Error):   {metrics['mape']:.2f}%")
        print("=" * 70)
        
        # Print to file - ensure it's in the same directory as submission
        submission_abs = os.path.join(repo_root, submission_file) if not os.path.isabs(submission_file) else submission_file
        log_file = submission_abs.replace('.csv', '_score.txt')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w') as f:
            f.write("MOVIE LENS RATING PREDICTION - SCORING RESULTS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Submission: {submission_file}\n")
            f.write(f"Predictions evaluated: {metrics['count']}\n")
            f.write(f"\nRMSE: {metrics['rmse']:.4f}\n")
            f.write(f"MAE:  {metrics['mae']:.4f}\n")
            f.write(f"MAPE: {metrics['mape']:.2f}%\n")
        
        print(f"\n💾 Score saved to: {log_file}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
