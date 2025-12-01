# Security Incident Report - API Key Leak

## Issue
A Google Gemini API key was accidentally committed to the repository in `config.yaml`.

**Leaked Key (now invalid):** `AIzaSyB4tnFzoHhEK2a8cg1iPW1aSqoEyMa0uNY`

**Commits affected:**
- `f62c8ac` - refactor: focus on indexing and retrieval only
- `90cb271` - feat: gpu support; nre, re for extract entities and relationship

## Immediate Actions Required

### 1. Revoke the API Key Immediately
Go to Google Cloud Console and revoke this key:
https://console.cloud.google.com/apis/credentials

### 2. Clean Git History (if repo not pushed publicly yet)
```bash
# Use git-filter-repo to remove the key from history
pip install git-filter-repo

# Remove config.yaml from all commits
git filter-repo --path config.yaml --invert-paths --force

# Or use BFG Repo Cleaner (faster)
bfg --delete-files config.yaml
```

### 3. If Already Pushed to GitHub
- **Revoke the API key immediately** (most important!)
- Create a new API key
- Consider the old key compromised
- Force push cleaned history (if no collaborators have pulled)

## Prevention Measures Implemented

1. ✅ Added `config.yaml` to `.gitignore`
2. ✅ Created `config.yaml.example` with placeholder values
3. ✅ Removed actual `config.yaml` from git tracking
4. ✅ Updated documentation to use environment variables

## Best Practices Going Forward

### Use Environment Variables
```bash
export GEMINI_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

### Use .env files (NOT committed to git)
```bash
# .env (add to .gitignore)
GEMINI_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
```

### Load from environment in code
```python
import os
api_key = os.environ.get("GEMINI_API_KEY")
```

## Resources
- [GitHub: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [git-filter-repo](https://github.com/newren/git-filter-repo)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)

## Status
- ⚠️ **ACTION REQUIRED:** Revoke API key at Google Cloud Console
- ✅ Key removed from current files
- ⚠️ **ACTION REQUIRED:** Clean git history before public push
