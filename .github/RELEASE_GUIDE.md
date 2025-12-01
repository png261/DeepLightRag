# Release Guide for DeepLightRAG

This guide explains how to release a new version of DeepLightRAG to PyPI.

## Prerequisites

### 1. PyPI Account
- Create account at https://pypi.org
- Verify email address
- Enable 2FA (recommended)

### 2. PyPI API Token
1. Go to https://pypi.org/manage/account/
2. Scroll to "API tokens"
3. Click "Add API token"
4. Name: `deeplightrag-github-actions`
5. Scope: `Project: deeplightrag` (after first upload, use entire account for now)
6. Copy the token (starts with `pypi-`)

### 3. Test PyPI (Optional but Recommended)
1. Create account at https://test.pypi.org
2. Generate API token
3. Use for testing releases

### 4. GitHub Secrets
Add these secrets to your GitHub repository:

**Settings → Secrets and variables → Actions → New repository secret**

Required secrets:
- `PYPI_API_TOKEN` - Your PyPI API token
- `TEST_PYPI_API_TOKEN` - Your Test PyPI token (optional)

## Automated Release Process

### Option 1: Release via GitHub Releases (Recommended)

1. **Update Version**
   ```bash
   # Edit pyproject.toml
   version = "1.0.1"  # Bump version
   ```

2. **Update Changelog**
   ```bash
   # Edit CHANGELOG.md
   ## [1.0.1] - 2024-12-01
   ### Added
   - New feature X
   ### Fixed
   - Bug Y
   ```

3. **Commit Changes**
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore: Bump version to 1.0.1"
   git push origin main
   ```

4. **Create GitHub Release**
   - Go to: https://github.com/YOUR_USERNAME/DeepLightRag/releases
   - Click "Draft a new release"
   - Tag version: `v1.0.1`
   - Release title: `v1.0.1`
   - Description: Copy from CHANGELOG.md
   - Click "Publish release"

5. **Automatic Deployment**
   - GitHub Actions will automatically:
     - Build the package
     - Run tests
     - Publish to PyPI
   - Monitor: https://github.com/YOUR_USERNAME/DeepLightRag/actions

### Option 2: Manual Trigger (Test First)

1. **Test on Test PyPI**
   - Go to: Actions → Publish to PyPI → Run workflow
   - Check "Upload to Test PyPI"
   - Click "Run workflow"
   - Wait for completion
   - Test install: `pip install -i https://test.pypi.org/simple/ deeplightrag`

2. **Deploy to Production**
   - Go to: Actions → Publish to PyPI → Run workflow
   - Uncheck "Upload to Test PyPI"
   - Click "Run workflow"

## Manual Release Process

If you prefer to release manually:

```bash
# 1. Clean previous builds
rm -rf build/ dist/ *.egg-info src/*.egg-info

# 2. Install build tools
pip install --upgrade build twine wheel

# 3. Build package
python -m build

# 4. Check package
twine check dist/*

# 5. Upload to Test PyPI (optional)
twine upload --repository testpypi dist/*

# 6. Test installation
pip install -i https://test.pypi.org/simple/ deeplightrag

# 7. Upload to PyPI
twine upload dist/*
```

## Version Numbering

Follow Semantic Versioning (SemVer):

- **Major** (1.0.0 → 2.0.0): Breaking changes
- **Minor** (1.0.0 → 1.1.0): New features, backward compatible
- **Patch** (1.0.0 → 1.0.1): Bug fixes, backward compatible

Examples:
- `1.0.0` - Initial release
- `1.0.1` - Bug fix
- `1.1.0` - New feature
- `2.0.0` - Breaking change

## Pre-Release Versions

For beta/alpha releases:

```bash
# In pyproject.toml
version = "1.1.0b1"  # Beta 1
version = "1.1.0rc1" # Release candidate 1
```

Install with:
```bash
pip install --pre deeplightrag
```

## Checklist Before Release

- [ ] Version bumped in `pyproject.toml`
- [ ] `CHANGELOG.md` updated
- [ ] All tests passing
- [ ] Documentation updated
- [ ] README.md reflects new version
- [ ] No uncommitted changes
- [ ] GitHub secrets configured
- [ ] Test on Test PyPI first (if major release)

## Post-Release Tasks

1. **Verify Installation**
   ```bash
   pip install --upgrade deeplightrag
   deeplightrag --version
   ```

2. **Update Documentation**
   - Update GitHub README if needed
   - Announce on social media
   - Update project website

3. **Monitor Issues**
   - Watch GitHub issues
   - Respond to installation problems
   - Collect feedback

## Troubleshooting

### Build Fails

**Issue:** Package build fails
```bash
# Check syntax errors
python -m py_compile src/deeplightrag/**/*.py

# Check pyproject.toml
python -m build --no-isolation
```

### Upload Fails

**Issue:** `403 Forbidden` error
- Check API token is valid
- Verify token has correct permissions
- Try regenerating token

**Issue:** Version already exists
```bash
# Bump version in pyproject.toml
# PyPI doesn't allow re-uploading same version
```

### Import Errors After Install

**Issue:** Can't import after installing
```bash
# Check installation
pip show deeplightrag

# Reinstall
pip uninstall deeplightrag
pip install --no-cache-dir deeplightrag
```

## GitHub Actions Workflows

### 1. `publish-to-pypi.yml`
- **Trigger**: Release published or manual
- **Purpose**: Build and publish to PyPI
- **Secrets**: `PYPI_API_TOKEN`, `TEST_PYPI_API_TOKEN`

### 2. `test.yml`
- **Trigger**: Push to main/develop, PRs
- **Purpose**: Test on multiple OS/Python versions
- **Matrix**: Ubuntu, Windows, macOS × Python 3.9-3.12

### 3. `lint.yml`
- **Trigger**: Push to main/develop, PRs
- **Purpose**: Code quality checks
- **Tools**: Black, Ruff, MyPy

## Emergency Procedures

### Yank a Release

If you need to remove a bad release:

```bash
# This hides it from pip install but keeps it available
# Go to: https://pypi.org/project/deeplightrag/
# Manage project → Releases → Options → Yank release
```

### Hotfix Release

For urgent fixes:

```bash
# Create hotfix branch
git checkout -b hotfix/1.0.2

# Make fixes
# ... edit files ...

# Bump version
# Edit pyproject.toml: version = "1.0.2"

# Commit and release
git commit -am "fix: Critical bug fix"
git push origin hotfix/1.0.2

# Create release on GitHub
# Or trigger manual workflow
```

## Support

- **Documentation**: Full docs in repository
- **Issues**: GitHub Issues tracker
- **Email**: nhphuong.code@gmail.com

---

**Good luck with your releases! 🚀**