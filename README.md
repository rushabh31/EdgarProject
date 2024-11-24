## Overview
This repository contains implementations of two tasks focused on document understanding and generative AI. Both tasks involve processing financial documents (10-K filings) to extract insights and provide actionable results. The repository is structured to include code, logs, results, and documentation for each task.

---

## **Task 1: Engineering**
### **Objective**
Develop a solution to visualize a set of documents in two-dimensional space, enabling users to identify clusters and outliers.

### **Dataset**
- **Year**: 2020
- **Filing Type**: 10-K
- **Sections**: All
- **Companies**: Limited to 10

### **Workflow**
1. **Preprocessing**:
   - Convert documents into manageable chunks.
   - Convert chunks into embeddings using a pre-trained embedding model.
   - Standard scale the embeddings to normalize features.

2. **Dimensionality Reduction**:
   - Perform Principal Component Analysis (PCA) for dimensionality reduction.
   - Reduce embeddings further for 2D visualization.

3. **Clustering and Outlier Detection**:
   - Apply K-means clustering and assign a cluster number to each chunk.
   - Identify outliers using statistical methods or distance metrics.

4. **Visualization**:
   - Generate plots to display embeddings in two dimensions:
     - Color-coded by assigned clusters.
     - Color-coded by outlier flags.
     - Color-coded by section numbers of the filings.

### **Deliverables**
- **Code**: Implementation of the entire pipeline from preprocessing to visualization.
- **Plots**:
  - Embeddings in 2 dimensions
  - Embeddings with clusters.
  - Embeddings with outliers.
  - Embeddings with section-based color coding.
- **Logs**: Step-by-step execution logs.
- **Results**: Plots.

---

## **Task 2: Generative AI**
### **Objective**
Demonstrate the ability of a language model to extract specific attributes from financial filings for a selected year using a structured output approach.

### **Dataset**
- **Year**: 2018–2020
- **Filing Type**: 10-K
- **Sections**: All
- **Company**: Focused on one company.
- **Attributes**: Extract 5 data attributes for a single year.

### **Workflow**
1. **Preprocessing**:
   - Convert documents into chunks.
   - Convert chunks into embeddings using a pre-trained embedding model.

2. **Query and Prompt Creation**:
   - Design a query targeting data extraction for the selected year.
   - Construct a prompt for extracting data attributes using generative AI.

3. **Validation**:
   - Create a validation dataset with five true values from document chunks.
   - Compare AI-generated outputs against validation data.

4. **Chunk Retrieval**:
   - Demonstrate that the LLM can retrieve relevant chunks for the specified year from the embedding object.

### **Deliverables**
- **Code**: Includes chunk processing, embedding generation, prompt design, and validation.
- **Results**: Extracted attributes compared against the validation dataset.
- **Logs**: Execution steps and key insights.

---

## **Project Structure**
```plaintext
├── code/
│   ├── task_1_engineering.py
│   ├── task_2_genai.py
├── logs/
│   ├── task_1_execution.log
│   ├── task_2_execution.log
├── results/
│   ├── task_1_engineering/
│   │   ├── Embeddings_in_2_Dimensions.png
│   │   ├── Embeddings_in_2D_Colored_by_Cluster.png
|   |   ├── Embeddings_in_2D_Colored_by_Outlier_Flag.png
│   │   └── Embeddings_in_2D_Colored_by_Section.png
│   │   └── Cluster_Visualization_with_Boundaries.png
│   │   └── Cluster_Visualization_with_Boundaries_Colored_by_Section.png
│   │   └── Readme.md
│   ├── task_2_genai/
│   │   ├── output.json
│   │   └── Readme.md
├── docs/
│   ├── task_1_engineering.md
│   ├── task_2_genai.md
│   └── setup_guide.md
├── .gitignore
├── Makefile
├── .ruff.toml
├── .pre-commit-config.yaml
└── README.md
```

---

## **Notes**
- For **Task 2**, we utilize **outlines for structured output** from the `langchain_community` LLMs. This is a new addition and requires installing the `langchain_community` module from the master branch:
  ```bash
  git+https://github.com/langchain-ai/langchain.git@master#subdirectory=libs/community
  ```
  Ensure the latest version is installed before running Task 2 to support structured outputs. 

---
