from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from palmerpenguins import load_penguins
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

penguins_df = load_penguins()
penguins_cleaned_df = penguins_df.dropna()

selected_features = ["bill_length_mm", "flipper_length_mm"]

model_params = {"n_clusters": 3,
                "max_iter": 300}

penguins_data_df = penguins_cleaned_df.copy()

features_df = penguins_data_df.loc[:, selected_features]

# Scale data ready for kmeans model
standard_scalar = StandardScaler()
features_standard_scalar = standard_scalar.fit_transform(features_df)

kmeans_model = KMeans(
    n_clusters=model_params["n_clusters"], max_iter=model_params["max_iter"])

penguin_predictions = kmeans_model.fit_predict(features_standard_scalar)
penguins_data_df.loc[:, "cluster"] = penguin_predictions

# Label cluster by the most common species found in the cluster
cluster_species = penguins_data_df.groupby(
    "species")["cluster"].agg(lambda x: x.value_counts().index[0])

cluster_species_df = cluster_species.reset_index()

# Insert cluster ids into data and then replace with the
# species label for pretty displaying
penguins_data_df['cluster'] = penguins_data_df['cluster'].astype(str)
cluster_species_df['cluster'] = cluster_species_df['cluster'].astype(str)

penguins_data_df.loc[:, "cluster"] = penguins_data_df["cluster"].map(
    cluster_species_df.set_index("cluster")["species"])


cols_compare = ["species",
                "cluster", "bill_length_mm", "flipper_length_mm"]

penguins_selected_df = penguins_data_df[cols_compare]

penguins_selected_df = penguins_selected_df.rename(columns={
    "species": "actual_species", "cluster": "predicted_species"})

penguins_confusion = pd.crosstab(penguins_selected_df['predicted_species'],
                                 penguins_selected_df['actual_species'])

penguins_confusion_mismatch = penguins_confusion.copy()

np.fill_diagonal(penguins_confusion_mismatch.values, 0)