# 🤖 GitHub Actions Setup Guide

This guide explains how to set up automated scoring for your MovieLens Rating Prediction challenge using GitHub Actions.

## What is GitHub Actions?

GitHub Actions is a **CI/CD (Continuous Integration/Continuous Deployment) platform** that automatically runs tasks when certain events happen in your repository (like pull requests, pushes, etc.).

For this challenge, we use it to:
- ✅ Automatically score submissions when users submit via pull requests
- ✅ Validate submission format
- ✅ Post scores as comments on PRs
- ✅ Update the leaderboard

## Setup Steps

### 1. Enable GitHub Actions

1. Go to your repository on GitHub
2. Click **Settings** → **Actions** → **General**
3. Under "Actions permissions", select **"Allow all actions and reusable workflows"**
4. Save

### 2. Workflow File is Already Created

The file `.github/workflows/score-submission.yml` is already in the repo. It will:
- Trigger on pull requests that modify files in `submissions/`
- Set up Python 3.10
- Install dependencies from `requirements.txt`
- Run the scoring script
- Post results as a PR comment

### 3. How Participants Submit

**Step-by-step for participants:**

```bash
# Clone the repo
git clone https://github.com/your-username/gnn.git
cd gnn

# Create a branch for their submission
git checkout -b submission/alice-v1

# Create their submission file
# File: submissions/alice_submission.csv
# Format: user_id, movie_id, rating

# Add and commit
git add submissions/alice_submission.csv
git commit -m "Submit: alice baseline model"

# Push to GitHub
git push origin submission/alice-v1

# Create Pull Request on GitHub
# - Go to github.com/your-repo
# - Click "New Pull Request"
# - GitHub Actions automatically runs!
# - View scores in PR comments
```

### 4. Automatic Workflow

When a PR is created with a new submission:

```
User creates PR
    ↓
GitHub detects .csv file change in submissions/
    ↓
GitHub Actions workflow triggers
    ↓
Python environment set up
    ↓
Dependencies installed (from requirements.txt)
    ↓
Scoring script runs: python scoring_script.py <file>
    ↓
Score posted as PR comment
    ↓
Participant sees their score!
```

### 5. View Workflow Runs

1. Go to your repo on GitHub
2. Click **Actions** tab
3. See all workflow runs with status (✅ passed, ❌ failed)
4. Click on a run to see logs

## File Structure

```
.github/
└── workflows/
    └── score-submission.yml          ← Workflow configuration

starter_code/
├── requirements.txt                  ← Python dependencies
├── scoring_script.py                 ← Scoring logic
└── update_leaderboard.py            ← Leaderboard manager

submissions/
├── rf_baseline_submission.csv        ← Example submission
└── rf_baseline_submission_score.txt  ← Automatically generated score

LEADERBOARD.md                        ← Auto-generated rankings
```

## Testing Locally First

Before relying on GitHub Actions, test scoring locally:

```bash
cd starter_code

# Test scoring script
python scoring_script.py ../submissions/rf_baseline_submission.csv

# Update leaderboard
python update_leaderboard.py
```

## Customizing the Workflow

### Change Python Version

Edit `.github/workflows/score-submission.yml`:

```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.11'  # Change here
```

### Add More Steps

Example: Send email notification after scoring

```yaml
- name: Send notification
  if: success()
  run: |
    python send_email.py
```

### Run Tests Before Scoring

```yaml
- name: Run tests
  run: pytest starter_code/tests/
```

## Troubleshooting

### Workflow not triggering?

- ✅ Check that file path matches: `submissions/*.csv`
- ✅ Confirm Actions are enabled in Settings
- ✅ Look at Actions tab for error logs

### Score file not generated?

- ✅ Check `test_ratings_hidden.csv` exists in `../data/`
- ✅ Verify submission CSV format
- ✅ Check GitHub Actions logs for errors

### Dependencies not installing?

- ✅ Update `starter_code/requirements.txt`
- ✅ Use pinned versions: `pandas==2.0.3` not just `pandas`

## Example PR Comment Output

When a submission is scored, GitHub automatically posts:

```
## 📊 Submission Scores

MOVIE LENS RATING PREDICTION - SCORING RESULTS
======================================================================
Submission: submissions/alice_submission.csv
Predictions evaluated: 20000

RMSE: 0.8234
MAE:  0.6521
MAPE: 15.30%
```

## Security Notes

- ⚠️ Keep `test_ratings_hidden.csv` private (add to `.gitignore` if needed)
- ⚠️ Don't expose ground truth in workflows
- ⚠️ Validate all submissions for malicious code

## Advanced: Custom Metrics

Edit `scoring_script.py` to add custom metrics:

```python
def compute_metrics(merged):
    # Add your metrics here
    custom_metric = calculate_something(y_true, y_pred)
    return {
        'rmse': rmse,
        'custom': custom_metric,
    }
```

## Need Help?

- GitHub Actions Docs: https://docs.github.com/en/actions
- YAML Syntax: https://yaml.org/
- Python GitHub API: https://github.com/actions/github-script

---

**That's it!** Your challenge is now automated! 🚀
