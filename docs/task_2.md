# **EDGAR Financial Analysis System**

---

## **Overview**

The **EDGAR Financial Analysis System** is a comprehensive solution for analyzing SEC 10-K filings using Retrieval-Augmented Generation (RAG). This project processes filings for a specific company across multiple years, generates embeddings, and answers user queries based on extracted data. The pipeline uses PySpark for data processing, HuggingFace models for embeddings, and LangChain for query handling.

---

## **Table of Contents**

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Detailed Explanation of the Pipeline](#detailed-explanation-of-the-pipeline)
6. [Usage](#usage)
7. [Results and Validation](#results-and-validation)
8. [Key Files](#key-files)
9. [Contributing](#contributing)
10. [License](#license)

---

## **System Architecture**

The system processes SEC filings as follows:

1. **Data Loading**: Loads 10-K filings for a specified company and years.
2. **Text Chunking**: Splits text into manageable chunks based on token limits.
3. **Embeddings**: Converts chunks into embeddings and stores them in a vector database.
4. **Query Execution**: Processes queries by retrieving relevant chunks and generating answers using a language model.
5. **Validation**: Validates the system by checking query results against known data points.

---

## **Installation**

### **Prerequisites**

1. Python 3.8+
2. Apache Spark 3.x
3. A GPU-compatible system for efficient embeddings.
4. Required Python libraries: HuggingFace Transformers, LangChain, PySpark, and datasets.

### **Steps**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/edgar-analysis-system.git
   cd edgar-analysis-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Apache Spark:
   - Download [Apache Spark](https://spark.apache.org/downloads.html).
   - Add Spark to your `PATH`.

---

## **Dataset**

The dataset consists of SEC 10-K filings (2018–2020) for a specific company. Filings include sections such as:
- **Management Discussion and Analysis**
- **Financial Statements**
- **Notes to Financial Statements**

We extract the following attributes:
- Revenue
- Net Income
- Number of Employees
- Deferred Tax Assets
- Quarterly Dividends

### **Data Source**
The dataset is fetched using the HuggingFace `load_dataset` function from the `"eloukas/edgar-corpus"` dataset.

---

## **Detailed Explanation of the Pipeline**

This section provides a complete step-by-step explanation of the pipeline, tied to the task steps.

---

### **Step 1: Convert Documents to Chunks**

#### **1.1 Load and Filter Datasets**

**Function:** `load_filtered_datasets`

- **What it does:** Loads SEC filings and filters them for a specific company and years.
- **Input:** 
  - `years`: List of years (e.g., `['2018', '2019', '2020']`).
  - `company_cik`: Unique identifier for the company (e.g., `320193` for Apple Inc.).
- **Output:** List of dictionaries containing:
  - Year
  - Section number
  - Section text.
- **Task Mapping:** This step retrieves the documents for processing.

---

#### **1.2 Create a Spark DataFrame**

**Function:** `create_spark_dataframe`

- **What it does:** Converts the filtered dataset into a structured Spark DataFrame.
- **Input:** List of dictionaries from `load_filtered_datasets`.
- **Output:** Spark DataFrame with:
  - `year`: Filing year.
  - `section_number`: Section ID.
  - `section_text`: Text content.
- **Task Mapping:** Structures the data for further processing.

---

#### **1.3 Calculate Chunking Requirements**

**Function:** `calculate_chunking_requirements`

- **What it does:** Analyzes each section to determine if chunking is necessary based on token limits.
- **Input:** 
  - Spark DataFrame from `create_spark_dataframe`.
- **Output:** Dictionary with chunking specifications:
  - Whether chunking is required.
  - Suggested chunk size, overlap, and number of chunks.
- **Task Mapping:** Prepares documents for token-limited processing.

---

#### **1.4 Split Data into Chunks**

**Function:** `process_and_split_data`

- **What it does:** Splits text into chunks using a `RecursiveCharacterTextSplitter`.
- **Input:** 
  - Spark DataFrame.
  - Chunking requirements.
- **Output:** Processed Spark DataFrame with:
  - `text`: Chunked text.
  - `year`: Filing year.
  - `section_number`: Section ID.
  - `chunk_number`: Chunk index.
- **Task Mapping:** Finalizes chunked documents for embedding.

---

### **Step 2: Convert Chunks to Embeddings**

#### **2.1 Create Vector Store**

**Function:** `create_vector_store`

- **What it does:** Converts chunks into embeddings using HuggingFace and stores them in a vector database.
- **Input:**
  - Processed DataFrame with chunks.
  - Embedding model name (e.g., `"sentence-transformers/all-mpnet-base-v2"`).
  - Directory to persist the database.
- **Output:** Vector database with embeddings and metadata.
- **Task Mapping:** Embeds chunks for query retrieval.

---

### **Step 3: Create a Query**

#### **3.1 Set Up LLM**

**Function:** `setup_llm`

- **What it does:** Configures a pre-trained language model for query handling.
- **Input:**
  - LLM model name (e.g., `"meta-llama/Llama-3.2-3B-Instruct"`).
- **Task Mapping:** Prepares the LLM for query generation.

---

### **Step 4: Create a Prompt**

#### **4.1 Year-Specific Query Pipeline**

**Function:** `create_year_specific_pipeline`

- **What it does:** Builds a pipeline to filter chunks by year, combine context, and format queries.
- **Input:**
  - Year (e.g., `2020`).
- **Output:** Query pipeline object.
- **Task Mapping:** Enables year-specific query handling.

---

### **Step 5: Validate Query Results**

#### **5.1 Run Queries**

**Function:** `run_queries`

- **What it does:** Processes user queries through the pipeline and retrieves relevant chunks.
- **Input:**
  - Query pipeline.
  - List of queries.
- **Output:** Results with answers and associated chunks.
- **Task Mapping:** Demonstrates query retrieval and accuracy.

---

### **Step 6: Save Results**

#### **6.1 Save Query Results**

**Function:** `save_query_results`

- **What it does:** Saves query results and metadata to a JSON file.
- **Input:**
  - Query results.
  - Output directory.
  - Company CIK.
- **Output:** JSON file with results.
- **Task Mapping:** Documents the system's performance.

---

### **Step 7: Cleanup**

**Function:** `cleanup`

- **What it does:** Stops the Spark session to free resources.
- **Task Mapping:** Cleans up after processing.

---

## **Usage**

### **Run the Pipeline**

1. Configure the system in the `main()` function:
   ```python
   CONFIG = {
       'years': ['2018', '2019', '2020'],
       'company_cik': '320193',  # Apple Inc.
       'embedding_model': "sentence-transformers/all-mpnet-base-v2",
       'llm_model': "meta-llama/Llama-3.2-3B-Instruct",
       'persist_dir': "chroma_db",
       'analysis_year': '2020'
   }
   ```

2. Run the script:
   ```bash
   python edgar_analysis_system.py
   ```

3. Submit sample queries:
   ```python
   SAMPLE_QUERIES = [
       "How much revenue did Apple earn in 2020?",
       "What was the number of employees reported in 2019?"
   ]
   ```

---

## **Results and Validation**

Results include:
- **Answers:** JSON-formatted responses.
- **Chunks:** Metadata and content used to generate answers.

---

