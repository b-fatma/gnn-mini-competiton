# ✅ How to Verify GitHub Actions is Working

## Quick Verification (2 minutes)

Run this command to check your setup:

```bash
cd /home/b-fatma/gnn
./test_github_actions.sh
```

**Expected output:** 
- ✅ All checks should pass
- If scoring test fails, just install dependencies (see below)

---

## Step-by-Step Verification

### 1. **Verify Local Setup** ✓

```bash
# Check files exist
ls -la .github/workflows/score-submission.yml
ls -la starter_code/scoring_script.py
ls -la data/test_ratings_hidden.csv

# Should all show files exist
```

### 2. **Test Scoring Script Locally** 

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (one-time)
pip install pandas scikit-learn

# Test scoring
cd starter_code
python3 scoring_script.py ../submissions/rf_baseline_submission.csv
```

**Expected:** See RMSE, MAE, MAPE scores and `_score.txt` file created

### 3. **Verify on GitHub** 

After pushing, GitHub Actions works when:

**Check 1: Actions Tab Shows Green**
```
1. Go to: https://github.com/b-fatma/gnn-mini-competiton/actions
2. Look for: "Score Submission" workflow
3. Status should be: ✅ (green checkmark)
```

**Check 2: Test with a Pull Request**
```bash
# On your local machine:
git checkout -b test-submission
echo "test" > submissions/test.csv
git add submissions/test.csv
git commit -m "Test submission"
git push origin test-submission

# On GitHub.com:
1. Create Pull Request
2. Go to "Actions" tab in your repo
3. Watch workflow run (takes 1-2 minutes)
4. Should say: "Score Submission" ✅
```

**Check 3: PR Comments**
When workflow completes, you should see:
- Comment on PR with scores
- Example:
  ```
  ## 📊 Submission Scores
  
  RMSE: 0.8234
  MAE:  0.6521
  MAPE: 15.30%
  ```

---

## Troubleshooting

### ❌ GitHub Actions tab doesn't show "Score Submission"

**Fix:**
1. Push code to GitHub:
   ```bash
   git add .
   git commit -m "Add GitHub Actions"
   git push origin master
   ```

2. Enable in GitHub Settings:
   - Go to repo: https://github.com/b-fatma/gnn-mini-competiton
   - Click: **Settings**
   - Click: **Actions** 
   - Click: **General**
   - Select: **"Allow all actions and reusable workflows"**
   - Click: **Save**

3. Wait 30 seconds and refresh

### ❌ Workflow doesn't trigger on PR

**Fix:**
- The workflow only triggers when files in `submissions/*.csv` are modified
- Make sure you're committing a `.csv` file to the `submissions/` folder
- Branch name doesn't matter

### ❌ Workflow runs but fails

**Check:**
1. Click the failed workflow run
2. Click the "score" job
3. Look at the logs to see what went wrong
4. Common issues:
   - `test_ratings_hidden.csv` path wrong
   - Missing dependency in `requirements.txt`
   - CSV format mismatch

---

## Minimal Test Without GitHub

To test everything locally without pushing:

```bash
# 1. Install dependencies
cd ~/gnn
source venv/bin/activate
pip install pandas scikit-learn numpy

# 2. Run scoring script
cd starter_code
python3 scoring_script.py ../submissions/rf_baseline_submission.csv

# 3. Check output
ls -la ../submissions/*_score.txt
cat ../submissions/rf_baseline_submission_score.txt
```

---

## Files Involved

```
Your Repository:
├── .github/workflows/
│   └── score-submission.yml          ← GitHub Actions config
├── starter_code/
│   ├── requirements.txt               ← Dependencies
│   ├── scoring_script.py              ← Scoring logic
│   └── update_leaderboard.py         ← Leaderboard manager
└── submissions/
    └── *.csv                          ← Participant submissions
```

When someone creates a PR with a `.csv` file:
1. GitHub detects the file change
2. Runs `score-submission.yml` workflow
3. Installs dependencies from `requirements.txt`
4. Runs `scoring_script.py`
5. Posts comment with results

---

## What Should Happen

**User Flow:**
```
Participant creates PR with submissions/alice.csv
    ↓
GitHub detects change in submissions/*.csv
    ↓
Workflow triggers automatically
    ↓
Python environment set up + dependencies installed
    ↓
scoring_script.py runs automatically
    ↓
Comment posted on PR with scores
    ↓
Participant sees their RMSE/MAE/MAPE instantly!
```

---

## Quick Commands

```bash
# Check if workflow file exists and has correct syntax
cat .github/workflows/score-submission.yml

# Test locally
source venv/bin/activate
pip install pandas scikit-learn
cd starter_code
python3 scoring_script.py ../submissions/rf_baseline_submission.csv

# Push to GitHub
git add .
git commit -m "GitHub Actions setup complete"
git push origin master

# Create test PR
git checkout -b test
echo "test" > submissions/test.csv
git add submissions/test.csv
git commit -m "Test"
git push origin test
# Then create PR on GitHub.com
```

---

## Summary

✅ **Local setup is working** (verified by test_github_actions.sh)

Next steps:
1. Push to GitHub
2. Enable Actions in Settings
3. Test with a PR submission
4. Watch Actions tab for automatic scoring

You're all set! 🚀
