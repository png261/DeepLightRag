# GitHub Automation for DeepLightRAG

This directory contains GitHub Actions workflows and documentation for automated testing and publishing.

## 📁 Contents

### Workflows
- **`publish-to-pypi.yml`** - Automatically publish to PyPI when a release is created
- **`test.yml`** - Run tests on multiple OS/Python versions
- **`lint.yml`** - Code quality checks with Black, Ruff, and MyPy

### Documentation
- **`RELEASE_GUIDE.md`** - Complete guide for releasing new versions
- **`SETUP_SECRETS.md`** - Step-by-step guide to set up PyPI tokens

## 🚀 Quick Start

### First Time Setup

1. **Get PyPI API Token**
   ```
   1. Go to https://pypi.org/manage/account/
   2. Create API token
   3. Copy the token (starts with pypi-)
   ```

2. **Add to GitHub Secrets**
   ```
   Settings → Secrets and variables → Actions
   Add secret: PYPI_API_TOKEN
   ```

3. **Done!** Now releases are automatic.

### Creating a Release

1. **Update version** in `pyproject.toml`
2. **Update** `CHANGELOG.md`
3. **Commit and push** changes
4. **Create GitHub Release** with tag like `v1.0.1`
5. **Wait** for automatic deployment!

## 🔄 Workflows Explained

### 1. Publish to PyPI (`publish-to-pypi.yml`)

**Triggers:**
- ✅ When you create a GitHub Release
- ✅ Manual trigger from Actions tab

**What it does:**
1. Checks out code
2. Sets up Python 3.10
3. Installs build tools
4. Builds package
5. Checks package with twine
6. Publishes to PyPI (or Test PyPI)

**Secrets needed:**
- `PYPI_API_TOKEN` (required)
- `TEST_PYPI_API_TOKEN` (optional, for testing)

### 2. Tests (`test.yml`)

**Triggers:**
- ✅ Push to main/develop branches
- ✅ Pull requests

**What it does:**
- Tests on Ubuntu, Windows, macOS
- Tests Python 3.9, 3.10, 3.11, 3.12
- Checks imports work
- Tests CLI commands
- Verifies platform detection

**Matrix:** 4 OS × 4 Python = 16 test combinations

### 3. Lint (`lint.yml`)

**Triggers:**
- ✅ Push to main/develop branches
- ✅ Pull requests

**What it does:**
- Runs Black (code formatting)
- Runs Ruff (linting)
- Runs MyPy (type checking)

## 📊 Workflow Status

Check status at:
```
https://github.com/YOUR_USERNAME/DeepLightRag/actions
```

## 🎯 Release Workflow

```
Developer                GitHub Actions              PyPI
─────────                ──────────────              ────

1. Bump version
2. Update changelog
3. Commit & push
4. Create release ──────> Trigger workflow ──────> Publish
5. Done! ✅                Tests pass ✅              Live on PyPI ✅
```

## 🧪 Testing Before Release

### Option 1: Test PyPI (Recommended)

```bash
# Trigger manual workflow
Actions → Publish to PyPI → Run workflow
✅ Check "Upload to Test PyPI"
```

Then test installation:
```bash
pip install -i https://test.pypi.org/simple/ deeplightrag
```

### Option 2: Local Build

```bash
./build_package.sh
pip install dist/deeplightrag-*.whl
```

## 🔧 Configuration

### Workflow Files Location
```
.github/workflows/
├── publish-to-pypi.yml    # Publishing automation
├── test.yml               # Testing automation
└── lint.yml               # Code quality automation
```

### Secrets Location
```
GitHub Repository
└── Settings
    └── Secrets and variables
        └── Actions
            ├── PYPI_API_TOKEN           # Required
            └── TEST_PYPI_API_TOKEN      # Optional
```

## 📋 Checklist for Contributors

Before submitting a PR:
- [ ] Tests pass locally
- [ ] Code formatted with Black
- [ ] No linting errors (Ruff)
- [ ] Type hints added (MyPy)
- [ ] Documentation updated
- [ ] Changelog updated (if needed)

## 🐛 Troubleshooting

### Publish Workflow Fails

**Error: 403 Forbidden**
- Check `PYPI_API_TOKEN` is set correctly
- Verify token hasn't expired
- Regenerate token if needed

**Error: Version already exists**
- Bump version number in `pyproject.toml`
- PyPI doesn't allow re-uploading same version

**Error: Package name taken**
- Choose different package name
- Update in `pyproject.toml`

### Test Workflow Fails

**Import errors**
- Check `pyproject.toml` dependencies
- Verify package structure

**Platform-specific failures**
- Check platform detection code
- Review OS-specific dependencies

### Lint Workflow Fails

**Black formatting**
```bash
black src/
```

**Ruff errors**
```bash
ruff check src/ --fix
```

**MyPy errors**
```bash
mypy src/deeplightrag --ignore-missing-imports
```

## 📚 Further Reading

- **Release Guide**: See `RELEASE_GUIDE.md`
- **Secrets Setup**: See `SETUP_SECRETS.md`
- **GitHub Actions Docs**: https://docs.github.com/en/actions
- **PyPI Publishing**: https://packaging.python.org/

## 🆘 Getting Help

- **Issues**: https://github.com/YOUR_USERNAME/DeepLightRag/issues
- **Discussions**: https://github.com/YOUR_USERNAME/DeepLightRag/discussions
- **Email**: nhphuong.code@gmail.com

## 📝 Workflow Badges

Add these to your main README:

```markdown
[![PyPI](https://img.shields.io/pypi/v/deeplightrag)](https://pypi.org/project/deeplightrag/)
[![Tests](https://github.com/YOUR_USERNAME/DeepLightRag/workflows/Tests/badge.svg)](https://github.com/YOUR_USERNAME/DeepLightRag/actions)
[![Lint](https://github.com/YOUR_USERNAME/DeepLightRag/workflows/Lint/badge.svg)](https://github.com/YOUR_USERNAME/DeepLightRag/actions)
```

## 🎉 Success!

Once set up, releases are fully automated:
1. Create release on GitHub
2. Wait ~5 minutes
3. Package live on PyPI!

---

**Happy releasing! 🚀**