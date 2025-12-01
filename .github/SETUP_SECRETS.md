# Setting Up GitHub Secrets for PyPI Publishing

## Step-by-Step Guide

### 1. Get PyPI API Token

#### Production PyPI
1. Go to https://pypi.org/manage/account/
2. Login or create account
3. Verify your email
4. Scroll down to "API tokens"
5. Click **"Add API token"**
6. Fill in:
   - **Token name**: `deeplightrag-github-actions`
   - **Scope**: `Entire account` (first time) or `Project: deeplightrag` (after first upload)
7. Click **"Add token"**
8. **COPY THE TOKEN** (starts with `pypi-...`)
   - ⚠️ You can only see it once!
   - Save it somewhere safe temporarily

#### Test PyPI (Optional)
1. Go to https://test.pypi.org/manage/account/
2. Create separate account (different from production)
3. Follow same steps as above
4. Get token starting with `pypi-...`

### 2. Add Secrets to GitHub

1. Go to your repository on GitHub
2. Click **Settings** (top right)
3. In left sidebar, click **Secrets and variables** → **Actions**
4. Click **"New repository secret"**

#### Add Production Token
- **Name**: `PYPI_API_TOKEN`
- **Value**: Paste your PyPI token (the one starting with `pypi-...`)
- Click **"Add secret"**

#### Add Test Token (Optional)
- **Name**: `TEST_PYPI_API_TOKEN`
- **Value**: Paste your Test PyPI token
- Click **"Add secret"**

### 3. Verify Secrets

After adding, you should see:
```
PYPI_API_TOKEN
TEST_PYPI_API_TOKEN
```

✅ Secrets are encrypted and cannot be viewed after creation.

## Quick Visual Guide

```
Your Computer                GitHub                  PyPI
─────────────                ──────                  ────
                                
1. Create token  ────────────────────────>  PyPI.org
   on PyPI.org              

                             2. Add token
                                to GitHub
                                Secrets
                                
3. Push code     ────────>   GitHub Actions ────>   Publish
   or release                                       to PyPI
```

## Testing the Setup

### Test with Test PyPI First

1. **Trigger Manual Workflow**
   - Go to: **Actions** → **Publish to PyPI**
   - Click **"Run workflow"**
   - ✅ Check "Upload to Test PyPI"
   - Click **"Run workflow"**

2. **Wait for Completion**
   - Watch the workflow run
   - Should complete in ~2-3 minutes

3. **Verify on Test PyPI**
   - Go to: https://test.pypi.org/project/deeplightrag/
   - Package should appear!

4. **Test Installation**
   ```bash
   pip install -i https://test.pypi.org/simple/ deeplightrag
   python -c "from deeplightrag import DeepLightRAG; print('OK')"
   ```

### Production Release

Once testing is successful:

1. **Create GitHub Release**
   - Go to: **Releases** → **Draft a new release**
   - Tag: `v1.0.0`
   - Title: `v1.0.0`
   - Description: Release notes
   - Click **"Publish release"**

2. **Automatic Deployment**
   - GitHub Actions automatically runs
   - Package published to PyPI
   - Monitor in **Actions** tab

3. **Verify Installation**
   ```bash
   pip install deeplightrag
   deeplightrag --version
   ```

## Security Best Practices

### ✅ Do's
- Use API tokens, not passwords
- Set minimal scope (project-specific after first upload)
- Use different tokens for Test PyPI and Production
- Rotate tokens periodically
- Enable 2FA on PyPI account

### ❌ Don'ts
- Never commit tokens to repository
- Don't share tokens in issues/PRs
- Don't use same token across projects
- Don't store tokens in code

## Troubleshooting

### Issue: Workflow Fails with "403 Forbidden"

**Cause:** Invalid or expired token

**Solution:**
1. Generate new token on PyPI
2. Update GitHub secret
3. Re-run workflow

### Issue: Workflow Fails with "Token has no permission"

**Cause:** Token scope too restrictive

**Solution:**
1. For first upload: Use "Entire account" scope
2. After first upload: Can use "Project: deeplightrag" scope
3. Update token in GitHub secrets

### Issue: Can't Find Secrets in Workflow

**Cause:** Secrets not set or wrong name

**Solution:**
1. Verify secret names match workflow:
   - `PYPI_API_TOKEN`
   - `TEST_PYPI_API_TOKEN`
2. Check secrets in Settings → Secrets and variables → Actions

### Issue: Package Already Exists

**Cause:** Version number already used

**Solution:**
1. Bump version in `pyproject.toml`
2. Commit changes
3. Create new release with new version

## Environment-Specific Tokens

You can also use environment-specific secrets:

### Production Environment
```yaml
environment:
  name: production
  secrets:
    PYPI_API_TOKEN: ${{ secrets.PROD_PYPI_TOKEN }}
```

### Staging Environment
```yaml
environment:
  name: staging
  secrets:
    PYPI_API_TOKEN: ${{ secrets.TEST_PYPI_TOKEN }}
```

## Token Management

### Rotating Tokens

Should rotate every 6-12 months:

1. Generate new token on PyPI
2. Update GitHub secret
3. Delete old token from PyPI
4. Test with manual workflow

### Multiple Projects

For multiple projects:

1. Create project-specific tokens
2. Use different GitHub secrets
3. Or use different scopes

Example:
- `PYPI_TOKEN_PROJECT_A`
- `PYPI_TOKEN_PROJECT_B`

## Quick Reference

### Required Secrets
| Secret Name | Description | Where to Get |
|------------|-------------|--------------|
| `PYPI_API_TOKEN` | Production PyPI | https://pypi.org/manage/account/ |
| `TEST_PYPI_API_TOKEN` | Test PyPI (optional) | https://test.pypi.org/manage/account/ |

### Workflow Files
| File | Purpose | Trigger |
|------|---------|---------|
| `publish-to-pypi.yml` | Publish package | Release or manual |
| `test.yml` | Run tests | Push/PR |
| `lint.yml` | Code quality | Push/PR |

### Commands
```bash
# Test release locally
./build_package.sh

# Manual upload
twine upload dist/*

# Test installation
pip install deeplightrag
```

## Need Help?

- **PyPI Documentation**: https://pypi.org/help/
- **GitHub Actions**: https://docs.github.com/en/actions
- **Issues**: Open issue in repository
- **Email**: nhphuong.code@gmail.com

---

**Setup complete! 🎉 Ready to publish!**