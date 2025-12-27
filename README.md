# MAF-RL: Multi-Modal Actor–Critic Reinforcement Learning for Sequential Recommendation

This repository contains the official implementation of **MAF-RL**, a **Multi-Modal Actor–Critic Reinforcement Learning framework** for sequential recommendation.
The framework integrates heterogeneous user–item signals (behavioral sequences, textual semantics, visual features, and contextual information) within a unified reinforcement learning architecture, enabling robust long-term preference modeling and improved recommendation quality.

The implementation is built on top of the **RecBole** framework to ensure reproducibility, scalability, and compatibility with standard recommendation benchmarks.

---

## 🔍 Overview

Modern recommender systems suffer from three key limitations:

1. Limited modeling of long-term user preferences
2. Inadequate fusion of multi-modal information
3. Optimization objectives misaligned with long-term user satisfaction

**MAF-RL** addresses these challenges by:

* Modeling recommendation as a **sequential decision-making process**
* Leveraging a **multi-modal fusion module** combining textual, visual, and behavioral signals
* Optimizing decisions via **Actor–Critic reinforcement learning (PPO)**
* Supporting **offline training** on real-world datasets such as MovieLens, Amazon, and Yelp

---

## 🧠 Core Contributions

* **Multi-Modal Fusion Module**
  Integrates textual, visual, and interaction-based embeddings through gated attention mechanisms.

* **Actor–Critic Architecture**
  A policy network (Actor) selects items, while a value network (Critic) estimates long-term reward.

* **Offline Reinforcement Learning**
  Learns from historical interaction logs without requiring online exploration.

* **RecBole-Compatible Design**
  Fully integrated with RecBole’s data pipeline, evaluation metrics, and configuration system.

---

## 📁 Repository Structure

```
MAF-RL/
├── data/                     # Dataset metadata and preprocessing scripts
├── recbole_config/           # YAML configs for datasets and experiments
├── maf_rl/
│   ├── recbole_ext/
│   │   ├── models/           # Actor, Critic, fusion modules
│   │   ├── trainer/          # PPO-based trainer
│   │   ├── dataloader/       # Sequential data handlers
│   │   └── utils/            # Reward, logging, helpers
│   └── scripts/              # Training & evaluation entry points
├── tests/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/RS-Research/MAF-RL.git
cd MAF-RL
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 Datasets

The framework supports standard sequential recommendation benchmarks:

* **MovieLens (1M / 20M)**
* **Amazon Review datasets**
* **Yelp**

Datasets should be preprocessed according to RecBole format.
Feature embeddings (text, image, graph) can be precomputed and loaded during training.

---

## 🚀 Running Experiments

Example training command:

```bash
python maf_rl/scripts/run_recbole_train.py \
  --config recbole_config/movielens1m.yaml
```

Evaluation metrics include:

* Hit Ratio (HR@K)
* NDCG@K
* Mean Reciprocal Rank (MRR)
* Long-term utility–based reward scores

---

## 📈 Model Highlights

* Sequential state modeling with temporal awareness
* Multi-source fusion via gated attention
* PPO-based policy optimization
* Compatible with offline recommendation settings
* Scalable to large datasets and long user histories

---

## 📄 Citation

If you use this code or build upon it, please cite our work:

```
@article{MAFRL2025,
  title={MAF-RL: Multi-Modal Actor–Critic Reinforcement Learning for Sequential Recommendation},
  author={Zare, Gholamreza and collaborators},
  journal={Under Review},
  year={2025}
}
```

---

## 📬 Contact

For questions, collaboration, or issues, please open a GitHub issue or contact the author directly.

---

This repository is intended for **research and academic use** and serves as a reproducible foundation for future work in reinforcement learning–based recommender systems.
