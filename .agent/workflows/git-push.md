---
description: Auto-commit and push with AI-generated commit message
---

# Git Push Workflow

When invoked, analyze the staged changes and push to remote with a relevant commit message.

## Steps

1. Check git status to see what files have changed:
   ```bash
   cd /home/lyju/EEGNEW && git status --short
   ```

2. View the diff to understand the changes:
   ```bash
   cd /home/lyju/EEGNEW && git diff --stat
   ```

3. Stage all changes:
   ```bash
   cd /home/lyju/EEGNEW && git add .
   ```

4. Based on the changes, generate a relevant commit message following this format:
   - First line: Brief summary (50 chars max), type prefix like `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`
   - Empty line
   - Optional body with bullet points explaining what changed

5. Commit with the generated message:
   ```bash
   git commit -m "<generated message>"
   ```

// turbo
6. Push to remote:
   ```bash
   git push
   ```

## Commit Message Types

- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code restructuring
- `docs:` Documentation changes
- `chore:` Maintenance tasks
- `style:` Formatting, no code change
- `test:` Adding tests
- `perf:` Performance improvement

## Example

For changes to model architecture:
```
feat: Add spike rate regularization to S-MSTT

- Added SpikeRateRegularizer for controlling firing sparsity
- Replaced mean-based skip with SpatialSkipConv
- Updated trainer to include spike loss in training
```
