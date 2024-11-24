## Overview

This task demonstrates the use of Generative AI (Gen AI) for extracting structured financial data from SEC 10-K filings. The system processes filings for Apple Inc. (2018-2020) and uses embeddings and a large language model (LLM) to retrieve specific attributes from the filings for the year 2020.

The task follows a structured workflow:
1. **Chunking and Embeddings**: Convert the filings into manageable text chunks and embed them.
2. **Query and Prompting**: Create queries and prompts to extract relevant information.
3. **Validation**: Verify the LLM's ability to retrieve correct chunks and answers for the specific year.

---

## Workflow

### 1. **Dataset**
- **Years**: 2018-2020
- **Filing Type**: 10-K
- **Company**: Apple Inc.
- **Attributes Extracted**: 
  - Total number of employees.
  - Year-over-year net sales increase (percentage and dollars).
  - iPhone sales performance compared to the previous year.
  - Net deferred tax assets and liabilities.
  - Quarterly cash dividend per share.

---

### 2. **Steps**

#### a. **Convert Documents to Chunks**
- The documents are segmented into chunks based on the model's token limit (e.g., 512 tokens) with overlap (10%) for contextual continuity.
- Chunk Metadata:
  - Year
  - Section Number
  - Chunk Number

#### b. **Convert Chunks to Embeddings**
- Each chunk is converted into an embedding using `sentence-transformers/all-mpnet-base-v2`.
- Embeddings are stored in a vector database (e.g., Chroma).

#### c. **Create a Query**
- Queries are designed to extract specific data attributes from the filings.
- Examples:
  - "How many total employees does Apple have?"
  - "How much did net sales increase in 2020?"
  - "Did iPhone sales increase or decrease compared to the previous year?"

#### d. **Create a Prompt**
- A specialized prompt instructs the LLM to focus on the given year's context and answer the query concisely:
  ```
  You are an assistant specialized in analyzing 10-K filings for the year {year}. Using the provided context, extract the precise answer and ignore external data.
  **Question:** {query}
  **Context:** {chunks}
  ```

#### e. **Validation Dataset**
- Five known true values from the dataset are used to validate the system's outputs:
  - Total employees: ~147,000.
  - Net sales increase: 6% or $14.3 billion.
  - iPhone sales decreased.
  - Deferred tax assets: $11.0 billion; liabilities: $2.8 billion.
  - Quarterly dividend: $0.205 per share.

#### f. **Retrieve Correct Chunks**
- The system validates that the retrieved chunks:
  - Match the year specified in the query.
  - Contain the correct context for answering the question.

---

## Results

### Queries and Outputs

1. **How many total employees does Apple have?**
   - **Answer**: ~147,000 employees.
   - **Validation**: Matches chunk metadata for 2020, Section 1.

2. **How much did net sales increase in 2020?**
   - **Answer**: Increase of 6% or $14.3 billion.
   - **Validation**: Matches chunk metadata for 2020, Section 7.

3. **Did iPhone sales increase or decrease in 2020?**
   - **Answer**: Decreased.
   - **Validation**: Matches chunk metadata for 2020, Section 7.

4. **What are the net deferred tax assets and liabilities?**
   - **Answer**: $11.0 billion (assets), $2.8 billion (liabilities).
   - **Validation**: Matches chunk metadata for 2020, Section 7.

5. **What was the quarterly cash dividend per share in 2020?**
   - **Answer**: $0.205 per share.
   - **Validation**: Matches chunk metadata for 2020, Section 7.

---

## Validation

To demonstrate correctness:
1. **Chunk Retrieval**:
   - The system retrieves chunks corresponding to the specific year (2020) and section.
   - Example:
     - Query: "How many total employees does Apple have?"
     - Retrieved Chunk:
       ```
       Employees
       As of September 26, 2020, the Company had approximately 147,000 full-time equivalent employees.
       ```
     - Metadata:
       - Chunk Number: 42
       - Section Number: 1
       - Year: 2020

2. **Answer Validation**:
   - The extracted answers align with the true values derived from the dataset.

---

## Outputs

### JSON Structure
Each query result includes:
- **Answer**: The specific response derived from the filings.
- **Chunks**: Supporting context with metadata.

Example Output:
```json
{
  "How many total employees does Apple have?": {
    "answer": "In the year 2020, Apple had approximately 147,000 full-time equivalent employees.",
    "chunks": [
      {
        "metadata": {
          "chunk_number": 42,
          "section_number": "1",
          "year": "2020"
        },
        "content": "Employees\nAs of September 26, 2020, the Company had approximately 147,000 full-time equivalent employees."
      }
    ]
  }
}
```

---

## Conclusion

This system demonstrates the power of Gen AI and RAG in extracting and validating structured data from complex financial documents, providing accurate and contextually relevant answers to user queries.
