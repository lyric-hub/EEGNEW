# Research Papers: Spiking Transformers for EEG Classification

> A curated list of papers for building a subject-independent SNN-based EEG classifier.

---

## 🔥 Core Spiking Transformer Papers

### 1. Spikformer: When Spiking Neural Network Meets Transformer
- **Venue**: ICLR 2023
- **Key Contribution**: First pure spiking transformer with Spiking Self-Attention (SSA)
- **arXiv**: https://arxiv.org/abs/2209.15425
- **Code**: https://github.com/ZK-Zhou/spikformer
- **Relevance**: Foundation architecture - SSA avoids softmax, uses spike-form Q/K/V

### 2. Spike-Driven Transformer (SDT)
- **Venue**: NeurIPS 2023
- **Key Contribution**: 87.2× lower energy than vanilla attention via mask+addition ops
- **Paper**: https://neurips.cc/virtual/2023/poster/72079
- **OpenReview**: https://openreview.net/forum?id=9FmolyOHi5
- **Relevance**: Pure spike-driven attention mechanism, ideal for neuromorphic hardware

### 3. Meta-SpikeFormer
- **Venue**: ICLR 2024
- **Key Contribution**: State-of-the-art ImageNet results with spiking transformer
- **OpenReview**: https://openreview.net/forum?id=bMGHxpnnZq
- **Relevance**: Shows spiking transformers can match/exceed ANNs

### 4. SpikeGPT: Generative Pre-trained Language Model with SNNs
- **Venue**: arXiv 2023
- **Key Contribution**: 260M parameter SNN language model, 22× more energy efficient
- **arXiv**: https://arxiv.org/abs/2302.13939
- **Code**: https://github.com/ridgerchu/SpikeGPT
- **Relevance**: Demonstrates scalability of SNNs to large-scale tasks

---

## 🧠 EEG + Spiking Neural Networks

### 5. ECA-ATCNet with SIT-Conversion for MI-EEG
- **Venue**: MDPI Electronics 2023
- **Key Contribution**: Converts transformer attention to SNNs for EEG-BCI
- **Results**: 87.89% (within-subject), 71.88% (between-subject) on BCI-2a
- **Link**: https://www.mdpi.com/journal/electronics (search: ECA-ATCNet SIT)
- **Relevance**: **Directly applicable** - MI-EEG + SNN + attention conversion

### 6. SpikeWavformer: Wavelet + Spiking Self-Attention for EEG
- **Venue**: Frontiers in Neuroscience 2024
- **Key Contribution**: Combines wavelet transform with spiking attention
- **Link**: https://www.frontiersin.org/journals/neuroscience
- **Relevance**: Handles non-smooth EEG signals, energy-efficient

### 7. Adaptive Graph SNN with LSTM for EEG Classification
- **Venue**: IEEE 2024
- **Key Contribution**: Graph convolution + LSTM for spatial-temporal EEG features
- **Relevance**: Captures electrode topology + temporal dynamics

---

## 🔄 Subject-Independent / Transfer Learning for EEG

### 8. EEGEncoder: Transformer for MI-EEG Classification
- **Venue**: arXiv 2024
- **Key Contribution**: Contextual processing with temporal dynamics for MI-EEG
- **arXiv**: https://arxiv.org/abs/2401.xxxxx (search: EEGEncoder motor imagery)
- **Relevance**: State-of-the-art transformer approach for comparison

### 9. ConvoReleNet: Subject-Independent MI Classification
- **Venue**: Frontiers in Neuroscience
- **Key Contribution**: Convolutional + relational modeling for cross-subject generalization
- **Link**: https://www.frontiersin.org/articles/10.3389/fnins.xxx
- **Relevance**: Architecture design for subject independence

### 10. Domain Adaptation for MI-BCI: A Survey
- **Venue**: Multiple (Frontiers, IEEE, NIH)
- **Topics**: Transfer learning, domain alignment, feature disentanglement
- **Relevance**: Essential for building subject-independent models

---

## 📚 Foundational / Background Papers

### 11. EEGNet: Compact CNN for EEG-based BCIs
- **arXiv**: https://arxiv.org/abs/1611.08024
- **Key Contribution**: Depthwise separable convolutions for efficient EEG classification
- **Relevance**: Baseline architecture, many SNN papers compare against this

### 12. BCI Competition IV Review (BNCI2014_001 Dataset)
- **Citation**: Tangermann et al., 2012, Frontiers in Neuroscience
- **DOI**: 10.3389/fnins.2012.00055
- **Relevance**: Your dataset's original publication

### 13. snnTorch: Deep Learning with Spiking Neural Networks
- **Docs**: https://snntorch.readthedocs.io/
- **Code**: https://github.com/jeshraghian/snntorch
- **Key Topics**: LIF neurons, BPTT, surrogate gradients
- **Relevance**: Your implementation framework

---

## 📥 Download Priority

| Priority | Paper | Why |
|----------|-------|-----|
| ⭐⭐⭐ | Spikformer | Core spiking attention architecture |
| ⭐⭐⭐ | Spike-Driven Transformer | Pure spike-driven ops for attention |
| ⭐⭐⭐ | ECA-ATCNet + SIT | Direct EEG + SNN + attention work |
| ⭐⭐ | SpikeGPT | Shows SNN scalability |
| ⭐⭐ | EEGNet | Baseline comparison |
| ⭐⭐ | Subject-independent papers | Cross-subject generalization |
| ⭐ | snnTorch tutorials | Implementation reference |

---

## 🔗 Quick Links

| Resource | URL |
|----------|-----|
| arXiv (preprints) | https://arxiv.org |
| OpenReview (ICLR/NeurIPS) | https://openreview.net |
| Frontiers | https://frontiersin.org |
| IEEE Xplore | https://ieeexplore.ieee.org |
| PapersWithCode | https://paperswithcode.com |
| snnTorch Tutorials | https://snntorch.readthedocs.io/en/latest/tutorials/index.html |
