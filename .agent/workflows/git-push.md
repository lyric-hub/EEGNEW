---
description: Auto-commit and push with AI-generated commit message (includes DVC)
---

# Git Push Workflow

When invoked, analyze the staged changes and push to remote with a relevant commit message.
This workflow also pushes DVC-tracked data before the Git push.

## Steps

1. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

2. Check git status to see what files have changed:
   ```bash
   git status --short
   ```

3. View the diff to understand the changes:
   ```bash
   git diff --stat
   ```

4. Stage all changes:
   ```bash
   git add .
   ```

5. Based on the changes, generate a relevant commit message following this format:
   - First line: Brief summary (50 chars max), type prefix like `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`
   - Empty line
   - Optional body with bullet points explaining what changed

6. Commit with the generated message:
   ```bash
   git commit -m "<generated message>"
   ```

// turbo
7. Push DVC-tracked data to remote storage:
   ```bash
   dvc push
   ```

// turbo
8. Push to Git remote:
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
