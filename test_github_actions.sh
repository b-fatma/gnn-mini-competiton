#!/bin/bash
# 🧪 GitHub Actions Setup Verification Script

echo "=================================================================="
echo "🧪 GITHUB ACTIONS SETUP VERIFICATION"
echo "=================================================================="

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0

# Helper functions
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅${NC} $2"
        ((PASS++))
        return 0
    else
        echo -e "${RED}❌${NC} $2 - FILE NOT FOUND: $1"
        ((FAIL++))
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✅${NC} $2"
        ((PASS++))
        return 0
    else
        echo -e "${RED}❌${NC} $2 - DIRECTORY NOT FOUND: $1"
        ((FAIL++))
        return 1
    fi
}

# 1. Check workflow file
echo ""
echo "1️⃣  CHECKING WORKFLOW SETUP"
echo "=================================================================="
check_dir ".github" "GitHub workflows directory exists"
check_dir ".github/workflows" "Workflows directory exists"
check_file ".github/workflows/score-submission.yml" "Score submission workflow"

# 2. Check required files
echo ""
echo "2️⃣  CHECKING REQUIRED FILES"
echo "=================================================================="
check_file "starter_code/requirements.txt" "Python requirements"
check_file "starter_code/scoring_script.py" "Scoring script"
check_file "starter_code/update_leaderboard.py" "Leaderboard manager"
check_file "data/test_ratings_hidden.csv" "Ground truth file"
check_file "data/train_ratings.csv" "Training data"

# 3. Check Git setup
echo ""
echo "3️⃣  CHECKING GIT SETUP"
echo "=================================================================="

if git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC} Git repository initialized"
    ((PASS++))
else
    echo -e "${RED}❌${NC} Not a Git repository"
    ((FAIL++))
fi

REMOTE=$(git remote get-url origin 2>/dev/null)
if [ ! -z "$REMOTE" ]; then
    echo -e "${GREEN}✅${NC} Remote configured: $REMOTE"
    ((PASS++))
else
    echo -e "${RED}❌${NC} No remote configured"
    ((FAIL++))
fi

BRANCH=$(git branch --show-current 2>/dev/null)
if [ ! -z "$BRANCH" ]; then
    echo -e "${GREEN}✅${NC} Current branch: $BRANCH"
    ((PASS++))
else
    echo -e "${RED}❌${NC} Could not determine current branch"
    ((FAIL++))
fi

# 4. Test scoring script
echo ""
echo "4️⃣  TESTING SCORING SCRIPT"
echo "=================================================================="

if [ -f "submissions/rf_baseline_submission.csv" ]; then
    echo "🔍 Testing with: submissions/rf_baseline_submission.csv"
    if python3 starter_code/scoring_script.py submissions/rf_baseline_submission.csv > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} Scoring script works correctly"
        ((PASS++))
    else
        echo -e "${RED}❌${NC} Scoring script failed"
        ((FAIL++))
    fi
else
    echo -e "${YELLOW}⚠️ ${NC}  No submission file to test with"
    echo "   To test: python3 starter_code/scoring_script.py submissions/your_file.csv"
fi

# 5. Workflow syntax (basic check)
echo ""
echo "5️⃣  CHECKING WORKFLOW FILE"
echo "=================================================================="

if grep -q "name: Score Submission" .github/workflows/score-submission.yml; then
    echo -e "${GREEN}✅${NC} Workflow file has correct name"
    ((PASS++))
else
    echo -e "${RED}❌${NC} Workflow name not found"
    ((FAIL++))
fi

if grep -q "pull_request:" .github/workflows/score-submission.yml; then
    echo -e "${GREEN}✅${NC} Workflow triggers on pull requests"
    ((PASS++))
else
    echo -e "${RED}❌${NC} Pull request trigger not found"
    ((FAIL++))
fi

if grep -q "submissions/\*.csv" .github/workflows/score-submission.yml; then
    echo -e "${GREEN}✅${NC} Workflow watches submissions directory"
    ((PASS++))
else
    echo -e "${RED}❌${NC} Submissions directory watch not found"
    ((FAIL++))
fi

# Summary
echo ""
echo "=================================================================="
echo "📊 SUMMARY"
echo "=================================================================="
echo -e "${GREEN}✅ Passed: $PASS${NC}"
echo -e "${RED}❌ Failed: $FAIL${NC}"

if [ $FAIL -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 All checks passed!${NC}"
    echo ""
    echo "NEXT STEPS:"
    echo "1. Push to GitHub:"
    echo "   git add ."
    echo "   git commit -m 'Set up GitHub Actions'"
    echo "   git push origin main"
    echo ""
    echo "2. Enable Actions in GitHub:"
    echo "   - Go to Settings → Actions → General"
    echo "   - Select 'Allow all actions and reusable workflows'"
    echo "   - Click Save"
    echo ""
    echo "3. Test with a PR:"
    echo "   - Create branch: git checkout -b test-submission"
    echo "   - Add a CSV file to submissions/"
    echo "   - Commit and push"
    echo "   - Create Pull Request on GitHub"
    echo "   - Watch Actions tab for automatic scoring!"
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}⚠️  Please fix the errors above${NC}"
    echo ""
    exit 1
fi
