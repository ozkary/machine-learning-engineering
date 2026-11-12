# Getting Started with Machine Learning Engineering

Welcome to the **Machine Learning Engineering** repository setup guide! This guide walks you through setting up your local environment, managing dependencies, launching interactive Jupyter notebooks, and testing containerized inference services.

---

## 1. System Requirements

Before you begin, ensure you have the following installed on your system:

- **Operating System:** Linux, macOS, or Windows with WSL2 (Windows Subsystem for Linux)
- **Python:** Version `3.9` or higher (`python3 --version`)
- **Git:** For version control and cloning (`git --version`)
- **Docker:** (Optional but recommended) For modules covering containerized deployment (`05-deployment`, `09-serverless`, `10-kubernetes`, and projects)
- **Pipenv:** For module-specific virtual environment management (`pip install pipenv`)

---

## 2. Clone the Repository

Clone the project to your local workstation:

```bash
git clone https://github.com/ozkary/ml-engineering.git
cd ml-engineering
```

---

## 3. Environment Setup Options

You can set up your Python environment using either a unified global virtual environment or module-specific environments.

### Option A: Unified Virtual Environment (Recommended for Notebooks)

To run notebooks across modules `01` through `04` and `06` through `08`:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Upgrade pip and install core data science packages
pip install --upgrade pip
pip install jupyterlab notebook numpy pandas scikit-learn matplotlib seaborn xgboost tensorflow
```

### Option B: Module-Specific Pipenv Environments (Recommended for Deployment)

Modules that include production code (e.g., `05-deployment`, `09-serverless`, `projects/heart-disease-risk`, `projects/drug-drug-interaction`) contain `Pipfile` and `Pipfile.lock` files for deterministic dependency resolution.

```bash
# Navigate to the module directory
cd 05-deployment

# Install exact pinned dependencies
pipenv install

# Activate the Pipenv shell
pipenv shell
```

---

## 4. Launching Jupyter Lab / Notebooks

Once your virtual environment is active, start Jupyter Lab from the repository root:

```bash
jupyter lab
```

This will launch a web interface in your browser (typically at `http://localhost:8888`). You can browse into any module folder and open the `homework.ipynb` or project notebooks.

---

## 5. Working with Containerized Services (Docker)

Several modules package trained models into containerized REST APIs.

### Example: Running the Module 05 Deployment Service

```bash
cd 05-deployment

# Build the Docker image
docker build -t ml-churn-service:v1 .

# Run the container locally mapping port 9696
docker run -it --rm -p 9696:9696 ml-churn-service:v1
```

In another terminal, test the prediction endpoint with a sample request:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"seniorcitizen": 0, "tenure": 1, "monthlycharges": 29.85, "totalcharges": 29.85}'
```

---

## 6. Recommended Learning Sequence

If you are new to machine learning:

1. Start with **[01-Introduction](./01-introduction/README.md)** to understand the core paradigm of learning from data.
2. Watch the companion video **[Discovering Machine Learning: A Primer Guide](https://www.youtube.com/live/ZJDyXNuHrfE?si=wxmU4sgTXlJGq1QZ)**.
3. Work through **[02-Regression](./02-regression/README.md)** and **[03-Classification](./03-classification/README.md)** to build models on real tabular datasets.
4. Dive into **[04-Evaluation](./04-evaluation/README.md)** to master cross-validation, precision-recall, and ROC-AUC metrics.
5. Advance to production engineering in **[05-Deployment](./05-deployment/README.md)**, **[06-Trees](./06-trees/README.md)**, **[08-Deep Learning](./08-deep-learning/README.md)**, and **[10-Kubernetes](./10-kubernetes/README.md)**.
6. Explore real-world capstone projects in **[projects/heart-disease-risk](./projects/heart-disease-risk/README.md)** and **[projects/drug-drug-interaction](./projects/drug-drug-interaction/README.md)**.

---

## 7. Need Help?

- Check existing discussions or file an issue on GitHub.
- Review the [Contribution Guidelines](./CONTRIBUTING.md).
- Follow updates on the [Newsletter](https://www.ozkary.com/newsletter) and [Twitter / X (@ozkary)](https://twitter.com/ozkary).
