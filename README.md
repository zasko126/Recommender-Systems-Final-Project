# Neural Networks for Recommendation Systems: Hybrid NCF + NLP 📚🤖

**Universidad de La Sabana — Faculty of Engineering** **Course:** Recommender Systems (2026.1)  
**Author:** Manuel Alejandro Niño Leon  

## 📌 Project Overview
This repository contains a complete, reproducible Deep Learning recommendation pipeline built with **PyTorch**. The project implements a **Hybrid Recommender System** that fuses two distinct modalities:
1. **Neural Collaborative Filtering (NCF):** Captures user-item interaction signals and latent behavioral patterns.
2. **Natural Language Processing (NLP):** Processes book synopses (TF-IDF representations) through a deep projection network to address the cold-start problem and enhance long-tail recommendations.

The architecture strictly follows modern representation learning practices, combining Generalized Matrix Factorization (GMF), a Multi-Layer Perceptron (MLP) tower, and a Content Projection branch into a unified fusion head.

## 📂 Repository Structure
- `Recommender_Final_Project.ipynb`: The main Jupyter Notebook containing the entire pipeline (Data Engineering, Architecture, Training, Evaluation, and Critical Analysis).
- `requirements.txt`: Python dependencies required to reproduce the environment.
- `README.md`: Project documentation and setup instructions.
- `eda_plots.png`, `embedding_space.png`, `learning_curves.png`, `topk_metrics.png`: Visualizations of the exploratory data analysis, latent space, training history, and evaluation metrics.

## 💾 Datasets & Download Instructions
This project merges two independent Kaggle datasets using fuzzy title normalization to create a robust, text-rich interaction matrix. 

Due to size limits, the datasets are not included in this repository. To run the notebook, please download the following files and place them in the root directory:
1. **Goodbooks-10k** ([Download Here](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k)):
   - Extract `ratings.csv`
   - Extract `books.csv` and **rename it** to `goodbooks_books.csv`
2. **GoodReads 100k** ([Download Here](https://www.kaggle.com/datasets/mdhamani/goodreads-books-100k)):
   - Extract the main CSV file and **rename it** to `books.csv`

## ⚙️ Setup & Dependencies
To replicate this environment, it is recommended to use a virtual environment (e.g., `venv` or `conda`). 

Install the required libraries using:
```bash
pip install -r requirements.txt
