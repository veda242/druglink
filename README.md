## 👥 Contributors
1. Sneha  
2. VedaSri  
3. Abhishek  
4. Praneeth  
5. Vineeth  

---

# 🧪 DrugLink  
**Interpretable Drug–Drug Interaction Prediction**

---

## 📌 Project Overview

DrugLink is a graph-based machine learning project designed to predict **drug–drug interactions (DDIs)** and explain **why those interactions occur**.

The project is inspired by the research paper:  
**“Accurate and Interpretable Drug–Drug Interaction Prediction Enabled by Knowledge Subgraph Learning (KnowDDI)”**.

Unlike traditional black-box models, DrugLink focuses on **interpretability**, making predictions more transparent and trustworthy for healthcare-related use cases.

---

## 🎯 Problem Statement

- There are thousands of drugs, and testing all possible drug combinations manually is not feasible.
- Many AI models can predict interactions but **do not explain their decisions**.
- Lack of explanation reduces trust and can be risky in healthcare.

**Goal of DrugLink:**  
To predict drug–drug interactions **accurately and explainably** using graph neural networks and knowledge graphs.

---

## 🧠 Base Research Paper

- **Paper:** Accurate and Interpretable Drug–Drug Interaction Prediction Enabled by Knowledge Subgraph Learning  
- **Model:** KnowDDI  

**Core Idea:**  
Combine drug–drug interaction data with biomedical knowledge graphs and use  
**Graph Neural Networks (GNNs)** to predict interactions and extract **interpretable knowledge subgraphs**.

---

## 🚀 Key Features

- Graph Neural Network–based DDI prediction  
- Knowledge subgraph extraction for interpretability  
- Dosage-aware feature extension  
- Confidence estimation for prediction reliability  
- Natural language explanation using LLM (**LLaMA via Groq**)

---

## 🧩 System Architecture (High Level)

1. User inputs two drug names (and dosage if available)  
2. Data is cleaned and preprocessed  
3. Drug–drug interaction data is merged with a biomedical knowledge graph  
4. GNN learns embeddings for drugs and biological entities  
5. A local knowledge subgraph is extracted for the drug pair  
6. Interaction prediction and probability are generated  
7. Confidence estimation and explanation are produced  

---

## 🛠️ Technology Stack

### Programming Language
- Python 3.x  

### Machine Learning & Graph Libraries
- PyTorch  
- DGL (Deep Graph Library)  

### Backend
- FastAPI  

### Database
- MongoDB (stores predictions, metadata, and logs)  

### LLM for Explanation
- LLaMA (via Groq API)  

### Development Tools
- VS Code  
- Jupyter Notebook  
- Git & GitHub  

---

## 📐 Mathematical Foundation (Brief)

- Biomedical data is modeled as a **knowledge graph**  
- Graph Neural Networks update node embeddings using **message passing**  
- For each drug pair, a **local knowledge subgraph** is extracted  
- The subgraph representation is used for interaction prediction  
- The model is trained using a **classification loss function**  

---

## 📂 Project Structure

```text
DRUGLINK/
├─ README.md
├─ run_all.ps1
├─ test_model.py
├─ train_baseline.py
├─ DRUGLINK/                     # app workspace
│  ├─ package.json / package-lock.json
│  ├─ node_modules/
│  ├─ DrugLink_frontend/
│  │  ├─ druglink-backend/       # Node/Express API
│  │  └─ druglink-frontend/      # React app
│  └─ DRUGLINK-model-main/       # model + API + eval
│     ├─ main.py                 # FastAPI entrypoint
│     ├─ requirements.txt
│     ├─ models/                 # pretrained model
│     ├─ scripts/                # training scripts
│     ├─ data/                   # datasets (LFS pointer)
│     └─ reports/CSVs            # eval outputs & summaries
```
## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
Open Command Prompt / Terminal and run:
```bash
git clone https://github.com/veda242/druglink.git
cd druglink
```
### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
```
### 3️⃣ Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```
**Mac / Linux:**
```bash
source venv/bin/activate
```
### 4️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 5️⃣ Run the Model (Testing)
```bash
python test_model.py
```
### 6️⃣ Provide Input
```bash
When prompted, enter drug names, for example:

aspirin with ibuprofen
```
### 7️⃣ View Output
```bash
The system will display:

Drug–drug interaction prediction

Probability score

Confidence level

Explanation based on knowledge subgraph
```
## 🛑 Common Issues

- Module not found error: Activate the virtual environment

- Version conflict: Ensure correct Python version

- Slow execution: Expected due to graph processing

## ✅ Notes

- The project runs locally

- No cloud deployment required

- This setup is sufficient for Milestone-2 demonstration
