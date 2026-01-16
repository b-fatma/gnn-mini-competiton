"""
📋 LEADERBOARD MANAGER - MovieLens Rating Prediction Challenge

Updates the leaderboard based on submission scores.
Tracks all submissions and rankings.
"""

import pandas as pd
import os
from datetime import datetime
import json
import subprocess


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


class LeaderboardManager:
    """
    Manage challenge leaderboard
    """
    
    def __init__(self, leaderboard_file='leaderboard.json'):
        repo_root = get_repo_root()
        # Use absolute path to leaderboard file
        if not os.path.isabs(leaderboard_file):
            self.leaderboard_file = os.path.join(repo_root, leaderboard_file)
        else:
            self.leaderboard_file = leaderboard_file
        self.repo_root = repo_root
        self.load_leaderboard()
    
    def load_leaderboard(self):
        """
        Load existing leaderboard or create new
        """
        if os.path.exists(self.leaderboard_file):
            with open(self.leaderboard_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                'challenge': 'MovieLens Rating Prediction',
                'submissions': [],
                'last_updated': None
            }
    
    def save_leaderboard(self):
        """
        Save leaderboard to file
        """
        self.data['last_updated'] = datetime.now().isoformat()
        with open(self.leaderboard_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def add_submission(self, user, filename, rmse, mae, mape, timestamp=None):
        """
        Add new submission to leaderboard
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        submission = {
            'user': user,
            'filename': filename,
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'mape': round(mape, 2),
            'timestamp': timestamp
        }
        
        self.data['submissions'].append(submission)
        self.save_leaderboard()
        
        return submission
    
    def get_rankings(self, metric='rmse', limit=10):
        """
        Get top rankings by metric (rmse, mae, or mape)
        Lower is better for all metrics
        """
        if not self.data['submissions']:
            return []
        
        sorted_subs = sorted(
            self.data['submissions'],
            key=lambda x: x[metric]
        )
        
        return sorted_subs[:limit]
    
    def get_user_best(self, user):
        """
        Get user's best submission
        """
        user_subs = [s for s in self.data['submissions'] if s['user'] == user]
        
        if not user_subs:
            return None
        
        return min(user_subs, key=lambda x: x['rmse'])


def generate_markdown_leaderboard(leaderboard_manager, output_file='LEADERBOARD.md'):
    """
    Generate markdown leaderboard file
    """
    # Use absolute path from repo root if not absolute
    if not os.path.isabs(output_file):
        repo_root = get_repo_root()
        output_file = os.path.join(repo_root, output_file)
    
    rankings = leaderboard_manager.get_rankings('rmse', limit=20)
    
    markdown = """# 🏆 MovieLens Rating Prediction - Leaderboard

📅 Last Updated: {timestamp}

## 🥇 Top 20 Submissions (by RMSE)

| Rank | User | RMSE ↓ | MAE | MAPE | Submitted |
|------|------|--------|-----|------|-----------|
""".format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    for i, sub in enumerate(rankings, 1):
        timestamp = sub['timestamp'][:10] if sub['timestamp'] else 'N/A'
        markdown += f"| {i} | {sub['user']} | {sub['rmse']:.4f} | {sub['mae']:.4f} | {sub['mape']:.2f}% | {timestamp} |\n"
    
    markdown += """
## 📊 Metrics Explained

- **RMSE** (Root Mean Squared Error): Penalizes larger errors more heavily. Lower is better.
- **MAE** (Mean Absolute Error): Average absolute difference between predictions and ground truth. Lower is better.
- **MAPE** (Mean Absolute Percentage Error): Percentage error relative to true values. Lower is better.

## 📝 How to Submit

1. Create a new branch: `git checkout -b submission/<your-name>`
2. Place your predictions in: `submissions/<your-name>_submission.csv`
3. File format must have columns: `user_id, movie_id, rating`
4. Ratings must be in range [1.0, 5.0]
5. Create a Pull Request
6. GitHub Actions will automatically score your submission
7. Check the PR comments for your score

## 🎯 Good Luck! 🚀
"""
    
    with open(output_file, 'w') as f:
        f.write(markdown)
    
    print(f"✅ Leaderboard saved to {output_file}")


def main():
    """
    Update leaderboard from submissions
    Scores any CSV files that don't have score files yet
    """
    print("=" * 70)
    print("📋 UPDATING LEADERBOARD")
    print("=" * 70)
    
    repo_root = get_repo_root()
    lb = LeaderboardManager()
    
    # Scan submissions directory
    submissions_dir = os.path.join(repo_root, 'submissions')
    if not os.path.exists(submissions_dir):
        print(f"No submissions directory found: {submissions_dir}")
        return
    
    # Find all CSV submission files
    csv_files = [f for f in os.listdir(submissions_dir) if f.endswith('.csv')]
    print(f"\nFound {len(csv_files)} CSV submission files")
    
    # Score any CSV files that don't have score files
    for csv_file in csv_files:
        csv_path = os.path.join(submissions_dir, csv_file)
        score_file = csv_file.replace('.csv', '_score.txt')
        score_path = os.path.join(submissions_dir, score_file)
        
        if not os.path.exists(score_path):
            print(f"\n🎯 Scoring: {csv_file}")
            # Run scoring script
            try:
                result = subprocess.run(
                    ['python', 'starter_code/scoring_script.py', os.path.relpath(csv_path, repo_root)],
                    cwd=repo_root,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"   ✅ Scored successfully")
                else:
                    print(f"   ❌ Scoring failed")
                    print(result.stderr)
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"✓ Already scored: {csv_file}")
    
    # Now load all score files
    score_files = [f for f in os.listdir(submissions_dir) if f.endswith('_score.txt')]
    
    print(f"\n📊 Loading {len(score_files)} score files...")
    
    for score_file in score_files:
        filepath = os.path.join(submissions_dir, score_file)
        
        # Parse score file
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract metrics (simple parsing)
        rmse = mae = mape = None
        
        for line in content.split('\n'):
            if 'RMSE:' in line:
                rmse = float(line.split(':')[1].strip())
            elif 'MAE:' in line:
                mae = float(line.split(':')[1].strip())
            elif 'MAPE:' in line:
                mape = float(line.split(':')[1].split('%')[0].strip())
        
        if rmse is not None and mae is not None and mape is not None:
            # Extract user from filename (e.g., "alice_submission_score.txt" -> "alice")
            user = score_file.replace('_submission_score.txt', '').replace('_score.txt', '')
            submission_file = score_file.replace('_score.txt', '.csv')
            
            # Check if already in leaderboard - only add if not already present with same metrics
            existing = [s for s in lb.data['submissions'] if s['filename'] == submission_file]
            if not existing or (existing and existing[0]['rmse'] != rmse):
                lb.add_submission(user, submission_file, rmse, mae, mape)
                print(f"✅ Added: {user} | RMSE: {rmse:.4f}")
            else:
                print(f"✓ Already in leaderboard: {user}")
    
    # Generate markdown leaderboard
    generate_markdown_leaderboard(lb)
    
    # Print top 5
    print("\n" + "=" * 70)
    print("🏆 TOP 5 SUBMISSIONS")
    print("=" * 70)
    
    top5 = lb.get_rankings('rmse', limit=5)
    for i, sub in enumerate(top5, 1):
        print(f"{i}. {sub['user']:20} RMSE: {sub['rmse']:.4f}  MAE: {sub['mae']:.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
