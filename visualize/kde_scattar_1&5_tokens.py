import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from collections import deque
from sklearn.manifold import TSNE
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from scipy.spatial.distance import euclidean


def pca_to_tsne(x, y, n_components, ratio_samples=0.01):
    """
    Use PCA to downsize the data to 10 dimensions, then use t-SNE to downsize the data to 3 dimensions, and return the downsized data and labels.

    Parameters:
    x (list or np.array): input data in the shape (n_samples, n_features).
    y (list or np.array): labels, shape (n_samples,).
    n_samples (int): total number of samples after balancing.

    Returns:
    x_embedded (np.array): the reduced data, of shape (n_samples, 3).
    y_resampled (np.array): balanced labels, shape (n_samples,).

    """
    ros = RandomOverSampler(random_state=42)
    x_resampled, y_resampled = ros.fit_resample(x, y)

    indices = np.random.choice(len(y_resampled), size=int(ratio_samples*len(y_resampled)), replace=False)
    x_limited = x_resampled[indices]
    y_limited = y_resampled[indices]

    pca = PCA(n_components=n_components, random_state=42)
    x_pca = pca.fit_transform(x_limited)
    if n_components > 3:
        return x_pca, y_limited
    else:
        tsne = TSNE(n_components=n_components, random_state=42)
        x_embedded = tsne.fit_transform(x_pca)
    return x_embedded, y_limited


def visualize_3d(x_embedded, y_resampled):
    """
    Visualize the data after 3-dimensional dimensionality reduction.

    Parameters:
    x_embedded (np.array): dimensionality reduced data, shape (n_samples, 3).
    y_resampled (np.array): labeled, shape (n_samples,).

    Returns:
    None
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    text = ['Mismatch', 'Match']
    for label in np.unique(y_resampled):
        ax.scatter(x_embedded[y_resampled == label, 0],
                   x_embedded[y_resampled == label, 1],
                   x_embedded[y_resampled == label, 2],
                   label=f'{label}', alpha=0.5)

    # ax.set_title('t-SNE 3D Visualization (Balanced Data)')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.legend()
    plt.savefig('fig6.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_pair_grid(df, hue_column):
    """
    Plots pairwise relationships, including scatter plots for upper triangles, KDE plots for lower triangles, and KDE plots for diagonals.

    Parameters:
    df (pd.DataFrame): dataset
    hue_column (str): the name of the column used to distinguish between categories
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df[hue_column] = df[hue_column].astype('category')

    g = sns.PairGrid(df, hue=hue_column)
    g.map_upper(sns.scatterplot, alpha=0.5)
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.kdeplot, fill=True)
    g.add_legend()

    # g = sns.PairGrid(df, hue="species")
    # g.map_upper(sns.scatterplot)
    # g.map_lower(sns.kdeplot, fill=True)
    # g.map_diag(sns.histplot, kde_kws={"shade": True})
    # g.add_legend()
    plt.savefig('fig7.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()

def calculate_cluster_center_distances(cluster_centers):
    """
    Calculates the Euclidean distances between cluster centers and returns the mean and minimum of these distances.

    Parameters:
    cluster_centers (np.array): cluster centers, of shape (n_clusters, n_features).

    Returns:
    mean_distance (float): the mean distance between cluster centers.
    min_distance (float): the minimum value of the distance between cluster centers.
    """
    n_clusters = cluster_centers.shape[0]
    distances = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            distances.append(euclidean(cluster_centers[i], cluster_centers[j]))
    mean_distance = np.mean(distances)
    min_distance = np.min(distances)
    return mean_distance, min_distance


def read_json_files_in_outputs(dir_path):
    files_path = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                files_path.append(file_path)
    return files_path


analyse_past_5 = True
dimension = 3

if __name__ == '__main__':
    files_path = read_json_files_in_outputs('outputs_logits')
    if_mismatch_list = []
    top100_logits = []
    past_5_top100_logits = []

    for file_path in files_path:
        # print(file_path)
        if file_path.split('/')[-1] == 'overall_eval.json':
            continue
        outputs_list = []
        with open(file_path, "r") as json_file:
            for line in json_file:
                # print(line)
                outputs_list.append(json.loads(line))
        for outputs_data in outputs_list:
            past_5_queue = deque(maxlen=5)
            for i in range(len(outputs_data['slm_logits_prob'])):
                if i <= 5 and analyse_past_5:
                    past_5_queue.append(outputs_data['slm_logits_prob'][i][0][0])
                    continue
                text = ['Mismatch', 'Match']
                if_mismatch_list.append(text[outputs_data['if_match_now'][i]])
                top100_logits.append(outputs_data['slm_logits_prob'][i][0][0])
                past_5_queue.append(outputs_data['slm_logits_prob'][i][0][0])
                past_5_top100_logits.append(list(past_5_queue))
                # print(outputs_data.keys())
    print(len(if_mismatch_list))
    print(len(top100_logits))
    print(len(past_5_top100_logits[0]))

    if analyse_past_5 == False:
        top100_logits = np.array(top100_logits)
        if_mismatch_list = np.array(if_mismatch_list)
        x_embedded, y_resampled = pca_to_tsne(top100_logits, if_mismatch_list, dimension)
    else:
        past_5_top100_logits = np.array(past_5_top100_logits)
        if_mismatch_list = np.array(if_mismatch_list)
        n_samples, time_steps, n_features = past_5_top100_logits.shape
        past_5_top100_logits_flat = past_5_top100_logits.reshape(n_samples, time_steps * n_features)
        x_embedded, y_resampled = pca_to_tsne(past_5_top100_logits_flat, if_mismatch_list, dimension)

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

    kmeans = KMeans(n_clusters=2, random_state=42)
    y_kmeans = kmeans.fit_predict(x_embedded)
    cluster_centers = kmeans.cluster_centers_

    silhouette_avg = silhouette_score(x_embedded, y_kmeans)
    davies_bouldin = davies_bouldin_score(x_embedded, y_kmeans)
    ari = adjusted_rand_score(y_resampled, y_kmeans)
    mean_distance, min_distance = calculate_cluster_center_distances(cluster_centers)

    print("Silhouette Coefficient:", silhouette_avg)
    print("Davies-Bouldin Index:", davies_bouldin)
    print("Mean Cluster Center Distance:", mean_distance)
    print("Minimum Cluster Center Distance:", min_distance)
    print("Adjusted Rand Index:", ari)

    if dimension == 3:
        visualize_3d(x_embedded, y_resampled)

    df = pd.DataFrame(x_embedded, columns=[f'Feature {index}' for index in range(x_embedded.shape[1])])
    df['Label'] = y_resampled

    plot_pair_grid(df, hue_column='Label')
    # df = generate_mock_data()
    # plot_pair_grid(df, 'species')


