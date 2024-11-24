
## Overview

This task analyzes and visualizes document embeddings by reducing their dimensionality, clustering them, and highlighting outliers. The resulting visualizations provide insights into the structure of the document embeddings, relationships between clusters, and how sections of the documents are distributed.

## Visualizations

The following plots are generated during the analysis process. Each plot serves a unique purpose in understanding the data:

### 1. **Embeddings in 2 Dimensions**
   - **Filename**: `Embeddings_in_2_Dimensions.png`
   - **Description**: This plot visualizes the document embeddings reduced to two dimensions using PCA. It provides a general overview of how the embeddings are distributed in the 2D space, without any clustering or coloring applied.

---

### 2. **Embeddings Colored by Cluster**
   - **Filename**: `Embeddings_in_2D_Colored_by_Cluster.png`
   - **Description**: This plot colors the embeddings based on the clusters assigned by the K-Means algorithm. It helps identify how the clusters are distributed and their separability in the reduced 2D space.

---

### 3. **Embeddings Colored by Outlier Flag**
   - **Filename**: `Embeddings_in_2D_Colored_by_Outlier_Flag.png`
   - **Description**: Outliers are identified based on their distance from cluster centers. 
     - **Blue** points represent normal embeddings.
     - **Red** points represent outliers.
   - **Purpose**: Useful for detecting anomalies or embeddings that deviate significantly from the cluster distributions.

---

### 4. **Embeddings Colored by Section**
   - **Filename**: `Embeddings_in_2D_Colored_by_Section.png`
   - **Description**: Points are colored based on the section of the document they belong to. This plot helps identify how different sections of the documents are distributed and whether sections align with specific clusters.

---

### 5. **Cluster Visualization with Outliers and Boundaries**
   - **Filename**: `Cluster_Visualization_with_Boundaries.png`
   - **Description**: This plot includes:
     - Cluster boundaries calculated using Convex Hulls.
     - Outliers highlighted in red.
   - **Purpose**: Helps visualize the spatial extent of each cluster, highlighting their boundaries and identifying outliers in the dataset.

---

### 6. **Cluster Visualization with Boundaries Colored by Section**
   - **Filename**: `Cluster_Visualization_with_Boundaries_Colored_by_Section.png`
   - **Description**: Similar to the "Cluster Visualization with Outliers and Boundaries" plot but:
     - Points are colored based on their corresponding sections instead of their clusters.
     - Outliers are still highlighted in red.
   - **Purpose**: Shows how different sections of the documents align with the clusters and whether sections overlap across cluster boundaries.

---

## How to Interpret the Plots

- **Clusters**:
  - Clusters represent groups of similar embeddings.
  - Boundaries indicate the spatial extent of clusters in the PCA-reduced 2D space.
  
- **Outliers**:
  - Points marked as outliers lie far from their assigned cluster centers.
  - These points might require further investigation.

- **Sections**:
  - Section-based coloring provides insights into whether specific document sections dominate certain clusters or overlap across multiple clusters.
