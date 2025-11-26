<div align=center>
<h1> Learning Advanced Self-Attention for Linear Transformers in the Singular Value Domain</h1>

![GitHub Repo stars](https://img.shields.io/github/stars/hyowonwi/agf)
 [![arXiv](https://img.shields.io/badge/arXiv-2505.08516-b31b1b.svg)](arxiv.org/abs/2505.08516)
 [![alphaXiv](https://img.shields.io/badge/alphaXiv-ff0000.svg)](https://www.alphaxiv.org/abs/2505.08516) 
 [![alphaXiv](https://img.shields.io/badge/IJCAI_2025-cab4a3.svg)](https://www.ijcai.org/proceedings/2025/0730) 

<div>
      <a href="https://scholar.google.co.kr/citations?user=-foMLcAAAAAJ&hl=en" target="_blank"><b>Hyowon Wi</b></a><sup>1</sup>,
      <a href="https://www.jeongwhanchoi.com" target="_blank">Jeongwhan Choi</a><sup>1</sup>,
      <a href="https://sites.google.com/view/noseong" target="_blank">Noseong Park</a><sup>1</sup>,
    <div>
    <sup>1</sup>KAIST
    </div>
</div>
</div>

---

This is the official implementation of our IJCAI 2025 paper "Learning Advanced Self-Attention for Linear Transformers in the Singular Value Domain" (AGF).

## 📚 Related Works from Our Group

| Venue | Paper | Code |
|:-----:|:------|:----:|
| **NeurIPS 2024** | [Graph Convolutions Enrich the Self-attention in Transformers!](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5eceb48c3bc8b5d936c05ff8e2ece65e-Abstract-Conference.html) | [![GitHub](https://img.shields.io/github/stars/jeongwhanchoi/gfsa?style=social)](https://github.com/jeongwhanchoi/gfsa) |
| **ICML 2024** | [Polynomial-based Self-Attention for Table Representation Learning](https://proceedings.mlr.press/v235/kim24ae.html) | [![GitHub](https://img.shields.io/github/stars/JayoungKim408/TabPSA?style=social)](https://github.com/JayoungKim408/TabPSA) |
| **AAAI 2024** | [An Attentive Inductive Bias for Sequential Recommendation Beyond the Self-Attention](https://ojs.aaai.org/index.php/AAAI/article/view/28747) | [![GitHub](https://img.shields.io/github/stars/yehjin-shin/BSARec?style=social)](https://github.com/yehjin-shin/BSARec) |
| **ICLR 2024** | [Learning Flexible Body Collision Dynamics with Hierarchical Contact Mesh Transformer](https://openreview.net/forum?id=90yw2uM6J5) | [![GitHub](https://img.shields.io/github/stars/yuyudeep/hcmt?style=social)](https://github.com/yuyudeep/hcmt) |

---

## 📌 TL;DR

We propose Attentive Graph Filter (AGF), a novel self-attention mechanism that interprets attention as learning graph filters in the singular value domain from the perspective of directed graph signal processing (GSP). AGF achieves linear complexity O(nd²) while effectively leveraging both low and high-frequency information, outperforming existing linear Transformers on various benchmarks.

## 🏗️ Method Overview

<div align="center">
<img src="assets/overview.png" width="90%" alt="AGF Method Overview">
</div>

### Key Insight: Self-Attention is a Low-Pass Filter

**Theorem 1.** *Let $M = \text{softmax}(Z)$ for any matrix $Z \in \mathbb{R}^{n \times n}$. Then $M$ inherently acts as a low pass filter.*

This means vanilla self-attention attenuates high-frequency information, limiting the expressive power of Transformers.

### Our Solution: Attentive Graph Filter (AGF)

AGF directly learns graph filters in the singular value domain:
$$
AGF(X) = U(X) Σ(X) V(X)^{T} XW_v
$$
where:
- $U(X) = \rho(XW_u) \in \mathbb{R}^{n \times d}$ (left singular vectors)
- $Σ(X) = \sum_ k θ_k T_k(\text{diag}(\sigma(XW_s))) \in \mathbb{R}^{n \times d \times d}$ (filtered singular values)
- $V(X)^{T} = \rho((XW_v)^T) \in \mathbb{R}^{d \times n}$ (right singular vectors)

**Theorem 2.** *If the coefficient θₖ of a graph filter can have negative values and learned adaptively, the graph filter will pass low and high frequency signals appropriately.*


