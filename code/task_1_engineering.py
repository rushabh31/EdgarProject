import logging
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
)
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import os
import time
import datasets
from datasets import DatasetDict


class Config:
    """Configuration class to abstract out all input parameters."""

    def __init__(self):
        # Model configurations
        self.model_name: str = 'all-mpnet-base-v2'
        self.model_token_limit: int = 512
        self.overlap_ratio: float = 0.1
        self.batch_size: int = 32
        self.n_components: int = 2
        self.n_clusters: int = 20

        # Dataset configurations
        self.dataset_name: str = 'eloukas/edgar-corpus'
        self.dataset_config: str = 'year_2020'
        self.trust_remote_code: bool = True
        self.num_rows: int = 10  # Limit to first N rows

        # Output configurations
        self.timestamp: str = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir: str = f"experiment_{self.timestamp}"
        self.plots_dir: str = os.path.join(self.output_dir, "plots")
        self.log_file: str = os.path.join(self.output_dir, "experiment.log")

        # Spark configurations
        self.spark_app_name: str = "Document_Analysis"
        self.spark_driver_memory: str = "4g"
        self.spark_executor_memory: str = "4g"


class DocumentAnalyzer:
    def __init__(self, config: Config, spark_session: SparkSession = None, logger: logging.Logger = None):
        """Initialize the DocumentAnalyzer with configurations."""
        self.config = config
        self.spark = spark_session or SparkSession.builder \
            .appName(self.config.spark_app_name) \
            .config("spark.driver.memory", self.config.spark_driver_memory) \
            .config("spark.executor.memory", self.config.spark_executor_memory) \
            .getOrCreate()
        self.model = SentenceTransformer(self.config.model_name)  # Pre-trained embedding model
        self.logger = logger or logging.getLogger('DocumentAnalyzer')
        self.logger.info("Initialized DocumentAnalyzer with provided configurations.")

    def preprocess_data(self, dataset: DatasetDict) -> pd.DataFrame:
        """
        Preprocess the input Hugging Face DatasetDict:
        - Convert the 'train' split to a pandas DataFrame.
        - Filter rows where 'cik' is not null.
        - Limit to the first N rows as specified in the config.

        Args:
            dataset (DatasetDict): The Hugging Face DatasetDict object.

        Returns:
            pd.DataFrame: Preprocessed pandas DataFrame.
        """
        self.logger.info("Starting data preprocessing...")
        # Filter and limit rows using Dataset API
        filtered_dataset = dataset['train'].filter(lambda x: x['cik'] is not None)
        limited_dataset = filtered_dataset.select(range(self.config.num_rows))
        self.logger.info(f"Filtered dataset to first {self.config.num_rows} rows with non-null 'cik'.")

        # Convert to pandas DataFrame
        df = limited_dataset.to_pandas()
        df.to_csv('task_1.csv', index=False)
        self.logger.info(f"Converted dataset to pandas DataFrame with shape {df.shape}.")
        return df

    def chunk_documents(self, data, model_token_limit: int = None, overlap_ratio: float = None):
        """Dynamically split documents into chunks with overlap based on token limits."""
        model_token_limit = model_token_limit or self.config.model_token_limit
        overlap_ratio = overlap_ratio or self.config.overlap_ratio
        self.logger.info("Starting dynamic document chunking...")

        # Identify section columns
        section_cols = [col for col in data.columns if col.startswith('section_')]
        self.logger.info(f"Found {len(section_cols)} section columns for chunking.")

        def calculate_chunking(text: str, token_limit: int, overlap_ratio: float) -> List[str]:
            """Calculate chunks dynamically based on token limit and overlap ratio."""
            words = text.split()
            token_count = len(words)
            chunks = []

            if token_count <= token_limit:
                # No chunking required
                return [text]

            chunk_size = token_limit
            overlap = int(chunk_size * overlap_ratio)
            start = 0

            while start < len(words):
                end = start + chunk_size
                chunk = ' '.join(words[start:end])
                chunks.append(chunk)
                start = end - overlap  # Move forward with overlap

            return chunks

        schema = StructType([
            StructField("cik", StringType(), True),
            StructField("year", StringType(), True),
            StructField("section", StringType(), True),
            StructField("chunk_text", StringType(), True),
            StructField("chunk_id", IntegerType(), True)
        ])

        chunk_rows = []
        for row in data.collect():
            self.logger.info(f"Processing row with cik: {row['cik']}, year: {row['year']}")
            for section in section_cols:
                if row[section] and len(str(row[section])) > 0:
                    text = str(row[section])
                    chunks = calculate_chunking(text, model_token_limit, overlap_ratio)
                    self.logger.info(f"Created {len(chunks)} chunks for section {section}.")
                    for i, chunk in enumerate(chunks):
                        chunk_rows.append((row['cik'], row['year'], section, chunk, i))

        self.logger.info(f"Total chunks created: {len(chunk_rows)}")
        return self.spark.createDataFrame(chunk_rows, schema)

    def create_embeddings(self, chunk_df, batch_size: int = None):
        """Generate embeddings for chunks."""
        batch_size = batch_size or self.config.batch_size
        self.logger.info("Starting embedding generation...")
        chunks_data = chunk_df.collect()
        chunks = [row['chunk_text'] for row in chunks_data]
        self.logger.info(f"Total number of chunks: {len(chunks)}")

        all_embeddings = []
        total_batches = (len(chunks) - 1) // batch_size + 1
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.logger.info(f"Encoding batch {i // batch_size + 1}/{total_batches}")
            batch_embeddings = self.model.encode(batch)
            all_embeddings.extend(batch_embeddings)

        rows = []
        for i, row in enumerate(chunks_data):
            embedding_vector = Vectors.dense(all_embeddings[i])
            rows.append((
                row['cik'],
                row['year'],
                row['section'],
                row['chunk_text'],
                row['chunk_id'],
                embedding_vector
            ))

        schema = StructType([
            StructField("cik", StringType(), True),
            StructField("year", StringType(), True),
            StructField("section", StringType(), True),
            StructField("chunk_text", StringType(), True),
            StructField("chunk_id", IntegerType(), True),
            StructField("embeddings", VectorUDT(), True)
        ])

        self.logger.info("Completed embedding generation.")
        return self.spark.createDataFrame(rows, schema)

    def process_embeddings(
        self,
        embedding_df,
        n_components: int = None,
        n_clusters: int = None
    ):
        """Process embeddings: scale, reduce dimensionality, cluster, and detect outliers."""
        n_components = n_components or self.config.n_components
        n_clusters = n_clusters or self.config.n_clusters
        self.logger.info("Starting processing of embeddings...")

        # Standard scaling
        self.logger.info("Performing standard scaling of embeddings...")
        scaler = StandardScaler(
            inputCol="embeddings",
            outputCol="scaled_embeddings",
            withStd=True,
            withMean=True
        )
        scaler_model = scaler.fit(embedding_df)
        scaled_df = scaler_model.transform(embedding_df)
        self.logger.info("Scaling complete.")

        # PCA for dimensionality reduction
        self.logger.info(f"Performing PCA to reduce dimensions to {n_components}...")
        pca = PCA(
            k=n_components,
            inputCol="scaled_embeddings",
            outputCol="pca_features"
        )
        pca_model = pca.fit(scaled_df)
        pca_df = pca_model.transform(scaled_df)
        self.logger.info("PCA complete.")

        # KMeans Clustering
        self.logger.info(f"Clustering data into {n_clusters} clusters using KMeans...")
        kmeans = KMeans(
            k=n_clusters,
            featuresCol="pca_features",
            predictionCol="cluster"
        )
        kmeans_model = kmeans.fit(pca_df)
        clustered_df = kmeans_model.transform(pca_df)
        self.logger.info("Clustering complete.")

        # Calculate distances and identify outliers
        self.logger.info("Calculating distances to cluster centers and identifying outliers...")
        pca_features = np.array([row.pca_features.toArray() for row in clustered_df.collect()])
        clusters = np.array([row.cluster for row in clustered_df.collect()])
        centers = np.array(kmeans_model.clusterCenters())

        distances = [
            np.linalg.norm(pca_features[i] - centers[clusters[i]])
            for i in range(len(pca_features))
        ]
        threshold = np.mean(distances) + 2 * np.std(distances)
        outliers = [1 if d > threshold else 0 for d in distances]
        num_outliers = sum(outliers)
        self.logger.info(f"Identified {num_outliers} outliers out of {len(distances)} data points.")

        # Add distances and outliers to DataFrame
        result_df = clustered_df.withColumn("distance_to_center", F.lit(distances))
        result_df = result_df.withColumn("is_outlier", F.lit(outliers))

        self.logger.info("Processing of embeddings complete.")
        return result_df, pca_features, distances, outliers

    def visualize_and_save_results(
        self,
        pca_features: np.ndarray,
        clusters: List[int],
        outliers: List[int],
        section_labels: List[str]
    ):
        """Visualize and save results, including cluster boundaries, outliers, and other plots."""
        self.logger.info("Starting visualization and saving results...")
        # Create directories for saving results
        os.makedirs(self.config.plots_dir, exist_ok=True)
        self.logger.info(f"Created output directories at {self.config.output_dir}")

        # Create a DataFrame for visualization and saving
        viz_data = pd.DataFrame(pca_features, columns=['PC1', 'PC2'])
        viz_data['Cluster'] = clusters
        viz_data['Is_Outlier'] = outliers
        viz_data['Section'] = section_labels

        # Update the palette dynamically based on the number of clusters
        unique_clusters = viz_data['Cluster'].nunique()
        cluster_palette = sns.color_palette('viridis', n_colors=unique_clusters)

        # Define plot settings for existing visualizations
        plot_settings = [
            {
                'title': "Embeddings in 2 Dimensions",
                'filename': "Embeddings_in_2_Dimensions.png",
                'hue': None,
                'palette': None
            },
            {
                'title': "Embeddings in 2D: Colored by Cluster",
                'filename': "Embeddings_in_2D_Colored_by_Cluster.png",
                'hue': 'Cluster',
                'palette': cluster_palette
            },
            {
                'title': "Embeddings in 2D: Colored by Outlier Flag",
                'filename': "Embeddings_in_2D_Colored_by_Outlier_Flag.png",
                'hue': 'Is_Outlier',
                'palette': {0: 'blue', 1: 'red'}
            },
            {
                'title': "Embeddings in 2D: Colored by Section",
                'filename': "Embeddings_in_2D_Colored_by_Section.png",
                'hue': 'Section',
                'palette': 'tab20'
            },
            {
                'title': "Cluster Visualization with Outliers and Boundaries",
                'filename': "Cluster_Visualization_with_Boundaries.png",
                'hue': 'Cluster',
                'palette': cluster_palette
            },
            {
                'title': "Cluster Visualization with Boundaries Colored by Section",
                'filename': "Cluster_Visualization_with_Boundaries_Colored_by_Section.png",
                'hue': 'Section',
                'palette': 'tab20'
            }
            
        ]

        # Generate and save plots
        # Generate and save plots
        for plot in plot_settings:
            plt.figure(figsize=(15, 10))
            sns.scatterplot(
                x='PC1',
                y='PC2',
                hue=plot['hue'],
                data=viz_data,
                palette=plot['palette'],
                alpha=0.7
            )
            if plot['title'] in [
                "Cluster Visualization with Outliers and Boundaries",
                "Cluster Visualization with Boundaries Colored by Section"
            ]:
                # Overlay outliers for relevant plots
                outliers_data = viz_data[viz_data['Is_Outlier'] == 1]
                plt.scatter(
                    outliers_data['PC1'],
                    outliers_data['PC2'],
                    color='red',
                    label='Outliers',
                    edgecolor='black',
                    linewidth=0.7,
                    s=100  # Larger marker size for outliers
                )

                # Draw cluster boundaries
                from scipy.spatial import ConvexHull
                for cluster_id in viz_data['Cluster'].unique():
                    cluster_points = viz_data[viz_data['Cluster'] == cluster_id][['PC1', 'PC2']].values
                    if len(cluster_points) > 2:  # ConvexHull requires at least 3 points
                        hull = ConvexHull(cluster_points)
                        for simplex in hull.simplices:
                            plt.plot(
                                cluster_points[simplex, 0],
                                cluster_points[simplex, 1],
                                linestyle='--',
                                color='gray',
                                alpha=0.7
                            )

            plt.title(plot['title'])
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            if plot['hue']:
                plt.legend(title=plot['hue'], loc='best')
            else:
                plt.legend().remove()
            save_path = os.path.join(self.config.plots_dir, plot['filename'])
            plt.savefig(save_path)
            plt.close()
            self.logger.info(f"Saved plot: {save_path}")

        # Save embeddings and metadata as CSV
        embeddings_path = os.path.join(self.config.output_dir, "embeddings_metadata.csv")
        viz_data.to_csv(embeddings_path, index=False)
        self.logger.info(f"Saved embeddings and metadata to: {embeddings_path}")

        self.logger.info("Visualization and saving results completed.")
        return viz_data



# Main Execution
if __name__ == "__main__":
    # Initialize configuration
    config = Config()

    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('Main')
    logger.info("Starting the document analysis process.")

    # Load the dataset
    logger.info(f"Loading the dataset '{config.dataset_name}' with config '{config.dataset_config}'...")
    dataset = datasets.load_dataset(
        config.dataset_name,
        config.dataset_config,
        trust_remote_code=config.trust_remote_code
    )

    # Initialize Spark and DocumentAnalyzer
    logger.info("Initializing Spark session and DocumentAnalyzer...")
    spark = SparkSession.builder \
        .appName(config.spark_app_name) \
        .config("spark.driver.memory", config.spark_driver_memory) \
        .config("spark.executor.memory", config.spark_executor_memory) \
        .getOrCreate()
    analyzer = DocumentAnalyzer(config=config, spark_session=spark, logger=logging.getLogger('DocumentAnalyzer'))
    logger.info("Spark session and DocumentAnalyzer initialized.")

    # Preprocess Data
    df = analyzer.preprocess_data(dataset)
    logger.info("Dataset loaded and preprocessed.")

    # Convert to PySpark DataFrame
    logger.info("Converting pandas DataFrame to PySpark DataFrame...")
    spark_df = spark.createDataFrame(df)
    logger.info("Conversion complete.")

    # Analyze Documents
    logger.info("Starting document analysis...")
    chunk_df = analyzer.chunk_documents(
        spark_df,
        model_token_limit=config.model_token_limit,
        overlap_ratio=config.overlap_ratio
    )
    embedding_df = analyzer.create_embeddings(chunk_df, batch_size=config.batch_size)
    processed_df, pca_features, distances, outliers = analyzer.process_embeddings(
        embedding_df,
        n_components=config.n_components,
        n_clusters=config.n_clusters
    )
    logger.info("Document analysis complete.")

    # Visualization
    logger.info("Starting visualization of results...")
    sections = [row.section.replace("section_", "") for row in chunk_df.collect()]
    clusters = [row.cluster for row in processed_df.collect()]
    viz_data = analyzer.visualize_and_save_results(
        pca_features,
        clusters,
        outliers,
        sections
    )
    logger.info("Visualization complete.")

    logger.info("Document analysis process finished.")
