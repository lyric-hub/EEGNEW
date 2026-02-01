---
description: Summarize a research paper and add to docs/papers/README.md
---

# Summarize Paper Workflow

Use this workflow to summarize a PDF paper and add it to the project documentation.

---

## Usage

Invoke with the paper path:
```
/summarize-paper references/paper_name.pdf
```

---

## Steps

1. **Read the PDF** using view_file tool

2. **Extract key information**:
   - Title and venue
   - Key contribution (1 sentence)
   - Relevance to S-MSTT
   - Techniques to potentially adopt

3. **Add summary to `docs/papers/README.md`** using this template:

```markdown
### [Paper Title]
- **Venue**: [Conference/Journal Year]
- **Key Contribution**: [1 sentence summary]
- **Relevance**: [Why it matters for S-MSTT]
- **Techniques to Adopt**: [List of specific methods]
- **arXiv/Link**: [URL if available]
```

4. **Update `references/README.md`** to mark paper as summarized

---

## Example Output

```markdown
### Spikformer: When Spiking Neural Network Meets Transformer
- **Venue**: ICLR 2023
- **Key Contribution**: First pure spiking transformer with Spiking Self-Attention (SSA)
- **Relevance**: Foundation architecture for our SSA implementation
- **Techniques to Adopt**: Spike-form Q/K/V, membrane potential encoding
- **arXiv**: https://arxiv.org/abs/2209.15425
```
