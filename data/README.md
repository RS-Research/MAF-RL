# Data Directory

This directory contains dataset-related resources used in the **MAF-RL** framework.
All datasets follow the **RecBole data format** and are processed to support sequential, multimodal, and reinforcement learning–based recommendation.

---

## 📌 Supported Datasets

MAF-RL is evaluated on widely used sequential recommendation benchmarks:

### 1. MovieLens (1M)

* User–item interaction data with timestamps
* Used for evaluating sequential and long-term preference modeling

### 2. Amazon Reviews

* Category-level datasets (e.g., Books, Electronics)
* Includes rich textual metadata for multi-modal modeling

### 3. Yelp

* User–business interactions with contextual signals
* Suitable for testing long-term decision-making and diversity-aware policies

---

## 📂 Data Format Requirements

All datasets must be converted into **RecBole-compatible format**, typically consisting of:

### Interaction file (`*.inter`)

Contains user–item interaction logs.

Example fields:

```
user_id  item_id  timestamp
```

### Optional side information

Depending on the experiment, the following files may also be included:

* `*.user` — user attributes
* `*.item` — item metadata (category, brand, etc.)
* Precomputed feature files (e.g., `.npy`, `.pt`) for:

  * textual embeddings
  * visual embeddings
  * graph-based representations

---

## 🔄 Preprocessing Workflow

1. Download raw dataset from official source
2. Convert raw data into RecBole-compatible format
3. Store processed files under `data/processed/<dataset_name>/`
4. Reference the dataset in the corresponding YAML config file

Example:

```yaml
dataset: movielens1m
data_path: data/processed/
```

---

## ⚠️ Notes

* Raw datasets are **not included** in this repository due to licensing restrictions.
* Please ensure that file names and column definitions match those specified in the RecBole configuration.
* Feature extraction (text, image, graph) should be performed **offline** and stored as lookup tables for efficient training.

---

## 📚 Related Resources

* RecBole Documentation: [https://recbole.io](https://recbole.io)
* MovieLens Dataset: [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
* Amazon Review Data: [https://nijianmo.github.io/amazon/](https://nijianmo.github.io/amazon/)
* Yelp Dataset: [https://www.yelp.com/dataset](https://www.yelp.com/dataset)

---

This directory is intentionally modular to support **reproducibility**, **scalability**, and **multi-modal experimentation** within the MAF-RL framework.
