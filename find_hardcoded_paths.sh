#!/bin/bash
# find_hardcoded_paths.sh
# Run this in your project root to find hardcoded paths

echo "=========================================================="
echo "  Scanning for hardcoded paths in .py, .yaml, .xml files"
echo "=========================================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FOUND_ISSUES=0

echo ""
echo "[1/4] Searching for /home/username patterns..."
RESULT=$(grep -rn "/home/[a-zA-Z0-9_-]*/" --include="*.py" --include="*.yaml" --include="*.xml" . 2>/dev/null | grep -v ".git" | grep -v "venv" | grep -v "__pycache__")

if [ ! -z "$RESULT" ]; then
    echo -e "${RED}Found hardcoded /home/ paths:${NC}"
    echo "$RESULT"
    FOUND_ISSUES=$((FOUND_ISSUES + 1))
else
    echo -e "${GREEN}✓ No /home/ paths found${NC}"
fi

echo ""
echo "[2/4] Searching for specific usernames (drl-68, asr)..."
RESULT=$(grep -rn "drl-68\|/home/asr" --include="*.py" --include="*.yaml" --include="*.xml" . 2>/dev/null | grep -v ".git" | grep -v "Author:" | grep -v "Email:")

if [ ! -z "$RESULT" ]; then
    echo -e "${RED}Found user-specific paths:${NC}"
    echo "$RESULT"
    FOUND_ISSUES=$((FOUND_ISSUES + 1))
else
    echo -e "${GREEN}✓ No user-specific paths found${NC}"
fi

echo ""
echo "[3/4] Searching for absolute project paths..."
RESULT=$(grep -rn "Sim-to-Sim_Policy_Transfer" --include="*.py" --include="*.yaml" --include="*.xml" . 2>/dev/null | grep -v ".git" | grep -v "README" | grep -v "http" | grep -v "github.com" | grep "/home/\|/Users/")

if [ ! -z "$RESULT" ]; then
    echo -e "${YELLOW}Found absolute project paths (check if they're problematic):${NC}"
    echo "$RESULT"
    FOUND_ISSUES=$((FOUND_ISSUES + 1))
else
    echo -e "${GREEN}✓ No absolute project paths found${NC}"
fi

echo ""
echo "[4/4] Searching for common problematic patterns..."
RESULT=$(grep -rn "os.path.expanduser\|Path.home()\|~/" --include="*.py" . 2>/dev/null | grep -v ".git" | grep "/home/\|/Users/")

if [ ! -z "$RESULT" ]; then
    echo -e "${YELLOW}Found home directory usage (verify these are correct):${NC}"
    echo "$RESULT"
else
    echo -e "${GREEN}✓ Home directory paths look OK${NC}"
fi

echo ""
echo "=========================================================="
if [ $FOUND_ISSUES -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! No hardcoded paths found.${NC}"
else
    echo -e "${YELLOW}⚠️  Found $FOUND_ISSUES potential issue(s). Review above.${NC}"
    echo ""
    echo "To fix hardcoded paths, use:"
    echo "  - Relative paths: ../logs/model.pt"
    echo "  - pathlib: Path(__file__).parent / 'logs' / 'model.pt'"
    echo "  - Environment vars: os.environ['HOME']"
fi
echo "=========================================================="