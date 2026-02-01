---
description: Update improvements.md when implementing or completing model improvements
---

# Update Improvements Workflow

Use this workflow to track model improvements in `docs/improvements.md`.

---

## When Implementing a New Improvement

1. Move the improvement from "Next Priority" or "Future Research" to "In Progress":

```markdown
## 🔄 In Progress

### [Improvement Name]
- **Started**: [Date]
- **Branch**: [git branch if applicable]
- **Status**: [brief status]
```

2. After implementation is complete, move to "Completed" section:

```markdown
## ✅ Completed (v3.X)

| Fix | Description | Impact |
|-----|-------------|--------|
| **[Name]** | [Description] | [High/Medium/Low] - [Brief explanation] |
```

3. Remove the item from "Next Priority" or "Future Research"

---

## When Adding a New Improvement Idea

Add to the appropriate section in `docs/improvements.md`:

- **Next Priority (v3.3)**: High-impact, ready to implement
- **Future Research (v4.0+)**: Needs more investigation

Template:
```markdown
### [Number]. [Improvement Name]

**Problem**: [What issue does this solve]

**Solution**: 
- [Approach 1]
- [Approach 2]

**Resources**:
- [Links or references]
```

---

## Quick Commands

// turbo
View current improvements:
```bash
cat docs/improvements.md
```

// turbo
Check completed count:
```bash
grep -c "^\| \*\*" docs/improvements.md
```
