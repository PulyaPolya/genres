from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
import numpy as np
import pandas as pd
import os
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.cluster  import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import hdbscan
import pandas as pd
from tqdm import tqdm 
from collections import Counter 
from transformers import ASTModel, ASTConfig
from sklearn.metrics import pairwise_distances
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
import umap.umap_ as umap   # need to install umap-learn
from sklearn.manifold import TSNE
import numpy as np

import pandas as pd
from tqdm import tqdm
import random
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from permetrics import ClusteringMetric
import pandas as pd
import numpy as np
import re


ast_base = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
harmonix_metadata = pd.read_csv("dataset/metadata_harmonix.csv")



def load_data(layer=-1, path= ""):
    name = f"embeddings_beat_this_gtzan_layer{layer}.npz"
    data = np.load(os.path.join(path, name))
    lst = data.files
    assert len(lst) == 5556
    return data
def prepare_spectrograms(data, keys_to_use= None):
    lst = data.files
    if not keys_to_use:
         keys_to_use =lst.copy()
    spectrograms = {}
    for key in lst:
            if key in keys_to_use:
        #if not key.startswith("groove"):
                spectrograms[key] = torch.tensor(data[key], dtype=torch.float32)
    spectrograms_np = torch.stack([v.squeeze(0) for v in spectrograms.values()]).numpy()
    print(spectrograms_np.shape)
    true_labels = [i.split("___")[0]  for i in spectrograms.keys()]
    true_labels[0]
    return spectrograms_np, true_labels, spectrograms.keys()
def scale_data(spectrograms_np):
    scaler = StandardScaler()
    spectrograms_scaled = scaler.fit_transform(spectrograms_np)
    return spectrograms_scaled

def prepare_data(layer, path = "", keys_to_use= None):
    data = load_data(layer, path)
    spectrograms_np, true_labels, keys = prepare_spectrograms(data, keys_to_use)
    spectrograms_scaled = scale_data(spectrograms_np)
    return spectrograms_scaled, true_labels, keys

def prepare_raw_spectrograms(data):
    lst = data.files
    spectrograms = {}
    standard_shape = 1020
    for key in lst:
            sp= torch.tensor(data[key], dtype=torch.float32)
            if sp.shape[0] >standard_shape:
                    sp = sp[:standard_shape, :]
            elif sp.shape[0]< standard_shape:
                    pad = standard_shape - sp.shape[0]
                    #print(pad)
                    sp = np.pad(sp, ((0, pad), (0, 0)), mode="constant")
            spectrograms[key]   = torch.tensor(sp, dtype=torch.float32) 
            #print(sp.shape) 
    spectrograms_np = torch.stack([v.squeeze(0) for v in spectrograms.values()]).numpy()
    print(spectrograms_np.shape)
    true_labels = [i.split("___")[0]  for i in spectrograms.keys()]
    true_labels[0]
    return spectrograms_np, true_labels, spectrograms.keys()
def pool_time(X):
    mean = X.mean(axis=1)          
    std  = X.std(axis=1)           
    p25  = np.percentile(X, 25, axis=1) 
    p75  = np.percentile(X, 75, axis=1)  
    pooled = np.concatenate([mean, std, p25, p75], axis=1)  
    return pooled
def prepare_data_raw(layer, path):
    data =np.load(path)
    spectrograms_np, true_labels, keys = prepare_raw_spectrograms(data)
    print(spectrograms_np.shape)
    pooled = pool_time(spectrograms_np)         
    pooled = np.nan_to_num(pooled, nan=0.0) 
    print(pooled.shape)
    spectrograms_scaled = scale_data(pooled)
    return spectrograms_scaled, true_labels, keys

# checking that each cluster has at least threshold number of gtzan tracks
def form_hard_clusters(predicted_labels, keys):
    num_clusters = len(set(predicted_labels))
    clusters = {i: [] for i in range (num_clusters)}
    for i in range (len(predicted_labels)):
        #print(int(predicted_labels[i]))
        clusters[int(predicted_labels[i])].append(list(keys)[i])
    return clusters 
def check_gtzan_presence(clusters, return_sizes = False, threshold =80):
    gtzan_genres = [ "gtzan_pop", "gtzan_rock", "gtzan_metal",
           "gtzan_jazz", "gtzan_hiphop", "gtzan_classical", "gtzan_country", "gtzan_reggae", "gtzan_disco", "gtzan_blues"]
    gtzan_counts_per_cluster = {}
    for cluster, labels in clusters.items():
        gtzan_per_cluster = 0
        for label in labels:
            for gtzan_genre in gtzan_genres:
                if gtzan_genre in label.lower():
                    gtzan_per_cluster += 1
                    break
        gtzan_counts_per_cluster[cluster] = gtzan_per_cluster
        if gtzan_per_cluster < threshold:
            return False
    if return_sizes:
        return gtzan_counts_per_cluster#
    else: 
        return True


def form_clusters(spectrogram_keys, predicted_labels ):
    keys = list(spectrogram_keys)
    num_clusters = len(set(predicted_labels))
    clusters = {i: [] for i in range (num_clusters)}
    #keys = list(spectrograms.keys())
    for i in range (len(keys)):
     
        cluster = predicted_labels[i]
        clusters[cluster].append(keys[i])
    return clusters


def get_genre_cluster_distribution(clusters, spectrogram_keys, dataset = False, gtzan= False):
    mapping = {"asap":"Classical",  "jaah": "Jazz", "filosax": 
                    "Jazz", "Beatles": "Rock", "hiphop": "Hip-Hop", 'Classic Rock': "Rock",
                    "hip-hop": "Hip-Hop", "Bach": "Classical", "classic": "Classical", "Beethoven": "Classical", "Reggaeton" : "Reggae", 'Dance/Electronic': "Electronic", 
                    "Fugue": "Classical", "Indie Rock" : "Rock", "Dubstep": "Electronic", "Prog": "Rock", "Punk": "Rock", "Grunge": "Rock"}
    if dataset: 
        mapping = {}
        datasets = Counter([key.split("___")[0] for key in list(spectrogram_keys)])
        genres = datasets
    elif gtzan:
            genres = [ "gtzan_pop", "gtzan_rock", "gtzan_metal",
               "gtzan_jazz", "gtzan_hiphop", "gtzan_classical", "gtzan_country", "gtzan_reggae", "gtzan_disco", "gtzan_blues"]
    else:
        
        
    
        genres = ["Groove", "asap", "Classical", "jaah", "Jazz", "Pop", "filosax",   "Soul", "Alternative",
                   "HJDB",   "Rock", "Country", "Reggae",  "R&B", 'Reggaeton',  'Dance/Electronic','Hip-Hop', "Electronic", 
                   "Disco", "Blues", "hiphop", "Beatles", "Folk", "Metal", "World",'Classic Rock',
                     "Bach", "classic", "Beethoven", "Fugue", "Funk",'Prog','Punk','Reggae', 'Grunge', 'Rock', "Dubstep", "Indie Rock"] # "guitar", "ballroom", "Latin",
    
    num_items = []
   
    all_genres_mapped = set([mapping.get(genre, genre) for genre in genres])
    cluster_counts = {}
    for cluster, labels in clusters.items():
        mapped_genres = []
        for label in labels:
            picked = False
            if "harmonix" in label and not gtzan:
                title = label.split("___")[1]
                #print(title)
                try:
                    genre_harmonix = harmonix_metadata.loc[harmonix_metadata["File"] == title, "Genre"].values[0]
                    if genre_harmonix in genres:
                        genre = genre_harmonix
                    # if genre == "Pop":
                    #     genre = "Pop(harmonix)"
                    picked =False if  pd.isna(genre)  else True
                except:
                    print(f"problems with the file {title}")
                    picked = False
            else:
                for genre in genres:
                    if genre.lower() in label.lower():
                            picked = True
                            break
            if picked == True:
                if gtzan:
                    #print(genre)
                    genre = genre.split("_")[1]
                        #mapped_genres.append(genre.split("_")[1])
                mapped_genre = mapping.get(genre, genre)
                if "-" not in mapped_genre and "HJDB" not in mapped_genre and "&" not in mapped_genre and "/" not in mapped_genre:
                    mapped_genre = mapped_genre.capitalize()
                mapped_genres.append(mapped_genre)
                # if genre == "jazz" and cluster == 1:
                #      print(title)
            
            if picked == False and not gtzan:
                mapped_genres.append("Unknown")
        cluster_counts[cluster] = Counter(mapped_genres)
        print(f"cluster {cluster}: {Counter(mapped_genres)}")
        d = Counter(mapped_genres)
        num_items.append(sum(d[item] for item in d))
        print(sum(d[item] for item in d))
    return cluster_counts, num_items

def apply_genre_mapping(track):
    genres = [ "asap", "Classical", "jaah", "Jazz",  "filosax",   "Soul", "Alternative",
               "HJDB",   "Rock", "Country", "Reggae", "Latin", "R&B", 'Reggaeton',  'Dance/Electronic','Hip-Hop', "Electronic", 
               "Disco", "Blues", "hiphop", "Beatles", "Folk", "Metal", "World",'Classic Rock',
                 "Bach", "classic", "Beethoven", "Fugue", "Funk",'Prog','Punk', 'Pop-Rock','Reggae', 'Grunge', 'Rock', "Dubstep", "Indie Rock", "Cafe_Zimmermann", "Pop"]
    mapping = {"asap":"Classical",  "jaah": "Jazz", "filosax": 
            "Jazz", "Beatles": "Rock", "hiphop": "Hip-Hop", 'Classic Rock': "Rock",
            "hip-hop": "Hip-Hop", "Bach": "Classical", "classic": "Classical", "Beethoven": "Classical", "Reggaeton" : "Reggae", 'Dance/Electronic': "Electronic", 
            "Fugue": "Classical", "Indie Rock" : "Rock", "Dubstep": "Electronic", "Prog": "Rock", "Punk": "Rock", "Grunge": "Rock", "Chopin": "Classical", "Stravinsky": "Classical", "Cafe_Zimmermann": "Classical"}
    if "harmonix" in track:
                title = track.split("___")[1]
                #print(title)
                try:
                    genre_harmonix = harmonix_metadata.loc[harmonix_metadata["File"] == title, "Genre"].values[0]
                    #print(genre_harmonix)
                    if genre_harmonix in genres:
                        return genre_harmonix
                except Exception as e:
                        print(e)
    else:
        for genre in genres:
            if genre.lower() in track.lower():
                
                return mapping.get(genre, genre)
    return track.split("___")[0]


def form_clusters_c_means(num_clusters, u, keys, probability=0.35):
    # u: (num_clusters, num_items)
    clusters = {i: [] for i in range(num_clusters)}
    predicted_labels_cmeans = []
    double_tracks = {}
    single_tracks = {}

    for i in range(u.shape[1]):
        memberships = u[:, i]
        # get top-2 clusters by membership (stable tie-breaker: lower index first)
        top2_idx = list(np.argsort(-memberships))[:2]
        top2_vals = memberships[top2_idx]

        if top2_vals[1] >= probability and "gtzan" not in keys[i]:
            # store unordered pair so [1,2] == [2,1]
            a, b = sorted(top2_idx)
            clusters[a].append(keys[i])
            clusters[b].append(keys[i])
            double_tracks[keys[i]] = (a, b)          # tuple, ordered
            predicted_labels_cmeans.append(f"{a} + {b}")
        else:
            a = top2_idx[0]
            clusters[a].append(keys[i])
            single_tracks[keys[i]] = a               # store single as int
            predicted_labels_cmeans.append(str(a))

    return predicted_labels_cmeans, double_tracks, single_tracks, clusters


def get_reduction_names(algorithm,n_components, n_neighbors = None, perplexities= None, min_dists = None):
    if algorithm == "umap":
        reduction_names= [ f"{algorithm}_ngh{str(elem)}" for elem in n_neighbors] 
        reduction_names= [ f"{name}_dist{str(min_dist)}" for name in reduction_names for min_dist in min_dists] 
    elif algorithm == "t-SNE":
        reduction_names= [ f"{algorithm}_p{str(elem)}" for elem in perplexities] 
    elif algorithm == "PCA":
        reduction_names = [algorithm]
    reduction_names = [ f"{name}_comp{str(n_n)}"  for name in reduction_names for n_n in n_components]

    return reduction_names
def get_clustering_names (algorithm, linkages= None, selections = None, min_clusters= None, min_samples = None,  clusters= None):
    if algorithm == "Agglomerative":
        reduction_names =  [ f"{algorithm}_{linkage}" for linkage in linkages] 
    elif algorithm == "HDBSCAN":
         reduction_names =  [ f"{algorithm}_{selection}" for selection in selections]
         reduction_names =   [ f"{name}_min{min_cluster}" for min_cluster in min_clusters for name in reduction_names]
         reduction_names =   [ f"{name}_smpl{min_sample}" for min_sample in min_samples for name in reduction_names]
         return reduction_names
    else:
        reduction_names =[algorithm]

    reduction_names = [f"{name}_{num_clusters}" for name in reduction_names for num_clusters in clusters ]
    return reduction_names

def hdbscan_with_noise_reassignment(
    X,
    *,
    min_cluster_size=100,
    min_samples=20,
    cluster_selection_method="eom",
    metric="euclidean",
    assignment="centroid",   
):

    if assignment == "membership":
        hdbscan_kwargs = {"prediction_data": True, **hdbscan_kwargs}

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        metric=metric,
        **hdbscan_kwargs
    ).fit(X)

    labels = clusterer.labels_.copy()
    n_noise = list(labels).count(-1)
    noise_mask = labels == -1
    if not np.any(noise_mask):
        return labels, clusterer.labels_ +1, clusterer, n_noise  # nothing to reassign
    unique_clusters = np.unique(labels[~noise_mask])
    if unique_clusters.size == 0:
        predicted_labels = clusterer.labels_ +1
        return predicted_labels, clusterer.labels_ +1, clusterer, n_noise 

    if assignment == "centroid":
        centroids = []
        for cid in unique_clusters:
            centroids.append(X[labels == cid].mean(axis=0))
        centroids = np.vstack(centroids)

        D = pairwise_distances(X[noise_mask], centroids, metric=metric)
        nearest = D.argmin(axis=1)
        reassigned_ids = unique_clusters[nearest]

        labels_reassigned = labels
        labels_reassigned[noise_mask] = reassigned_ids
        predicted_labels = clusterer.labels_
        return labels_reassigned, predicted_labels, clusterer, n_noise

    elif assignment == "membership":
        try:
            mv = hdbscan.all_points_membership_vectors(clusterer) 
        except Exception as e:
            raise RuntimeError(
                "Membership assignment requires prediction_data=True and a successful fit."
            ) from e
        mv_argmax = mv.argmax(axis=1)

        existing_ids = np.unique(labels[labels != -1])
        existing_ids.sort()
        col_to_label = np.array(existing_ids)

        labels_reassigned = labels
        labels_reassigned[noise_mask] = col_to_label[mv_argmax[noise_mask]]
        return labels_reassigned, clusterer.labels_, clusterer, n_noise

    else:
        raise ValueError("assignment must be 'centroid' or 'membership'.")



def apply_reduction(name, n_components, seed, spectrograms_scaled, perplexity = None, n_neighbors = None, min_dist = None):
    if name.startswith("PCA"):
        pca = PCA(n_components=n_components, random_state=seed)
        spectrograms_reduced = pca.fit_transform(spectrograms_scaled)
    elif name.startswith("umap"):
        reducer = umap.UMAP(n_components =n_components,  n_neighbors =n_neighbors, min_dist = min_dist, random_state = seed)
        spectrograms_reduced = reducer.fit_transform(spectrograms_scaled)
    elif name.startswith("t-SNE"):
        spectrograms_reduced = TSNE(n_components=n_components, perplexity=perplexity, random_state=seed).fit_transform(spectrograms_scaled)
    elif  name.startswith("None"):
        return spectrograms_scaled
    return spectrograms_reduced

def cluster_points(name,spectrograms_reduced,  num_cluster = None, seed = 42, 
                   min_cluster_size =15, linkage = 'ward', selection = None, min_samples = None, eps = None, assign_noise = False):
    if name.startswith("K-means"):
        kmeans = KMeans(n_clusters = num_cluster,random_state=seed).fit(spectrograms_reduced)
        predicted_labels =kmeans.labels_

    elif name.startswith("Agglomerative"):
        agglomerative = AgglomerativeClustering(n_clusters=num_cluster, linkage = linkage).fit(spectrograms_reduced)
        predicted_labels = agglomerative.labels_
    elif name.startswith("Spectral"):
        spectral = SpectralClustering(n_clusters=num_cluster, assign_labels='discretize', random_state=42).fit(spectrograms_reduced)
        predicted_labels = spectral.labels_
    elif name.startswith("DBSCAN"):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(spectrograms_reduced)
        predicted_labels = dbscan.labels_
    elif name.startswith("HDBSCAN"):
        if assign_noise:
            predicted_labels, predicted_labels_old, clusterer = hdbscan_with_noise_reassignment(spectrograms_reduced, 
                                                                                              min_cluster_size =min_cluster_size, min_samples=min_samples, cluster_selection_method = selection)
        else:
            hdbscan = HDBSCAN(min_cluster_size =min_cluster_size,  cluster_selection_method = selection, min_samples=min_samples).fit(spectrograms_reduced)  
            predicted_labels = hdbscan.labels_+1  # treating noise as cluster
    return predicted_labels



def find_best_parameters_supervised (algorithms_names, reduction_names, spectrograms_scaled,  num_trials, keys):
    perplexity, n_neighbors, linkage, selection, min_cluster_size,  n_components, min_samples, min_dist   = None, None, None, None, None, None, None, None
    seeds = []
    metrics = [ "GTZAN", "Silhouette", "Seed", "CH", "Dunn"]
    cols = pd.MultiIndex.from_product([reduction_names, metrics], names=["Reduction", "Metric"])
    df = pd.DataFrame(index=algorithms_names, columns=cols)
    for r in df.index:
        for c in df.columns:
            df.at[r, c] = []
    patience_dict = {}

    for i in tqdm(range(num_trials)):
        seed = random.randint(0, 2**21-1)
        seeds.append(seed)
        for reduction in tqdm(reduction_names):
            if reduction.startswith("umap"):
                n_neighbors = int(reduction.split("_")[1][3:])
                min_dist = float(reduction.split("_")[2][4:])
                n_components = int(reduction.split("_")[3][4:])
            elif reduction.startswith("t-SNE"):
                perplexity = int(reduction.split("_")[1][1:])
                n_components = int(reduction.split("_")[2][4:])
            elif reduction.startswith("PCA"):
                n_components = int(reduction.split("_")[1][4:])
            
            spectrograms_reduced = apply_reduction(name= reduction, n_components= n_components, n_neighbors= n_neighbors,
                                                    seed= seed, spectrograms_scaled= spectrograms_scaled, perplexity=perplexity,  min_dist = min_dist)
            for algorithm in algorithms_names: 
                    current_patience = patience_dict.get((algorithm, reduction), 0) 
                    if current_patience < 3:
                        
                        if algorithm.startswith("HDBSCAN"):
                            selection = algorithm.split("_")[1]
                            min_cluster_size = int( algorithm.split("_")[2][3:])
                            min_samples = int( algorithm.split("_")[3][4:])
                            predicted_labels, n_noise= cluster_points(name= algorithm, spectrograms_reduced=spectrograms_reduced, num_cluster = num_cluster,
                                    seed = seed, min_cluster_size =min_cluster_size, linkage = linkage, selection = selection, min_samples = min_samples)
                        else:
                            if  algorithm.startswith("Agglomerative"):
                                linkage = algorithm.split("_")[1]
                            num_cluster =int(algorithm.split("_")[-1])
                            predicted_labels= cluster_points(name= algorithm, spectrograms_reduced=spectrograms_reduced, num_cluster = num_cluster,
                                        seed = seed, min_cluster_size =min_cluster_size, linkage = linkage, selection = selection, min_samples = min_samples)
                        clusters = form_hard_clusters(predicted_labels, keys)
                        gtzan_presence = check_gtzan_presence(clusters, return_sizes = False, threshold =80)
                        df.at[f"{algorithm}", (reduction, "GTZAN")].append( 1 if gtzan_presence else 0)
                        if gtzan_presence:
                            sil = np.nan
                            labels_for_sil = np.asarray(predicted_labels)
                            mask = labels_for_sil != -1
                            sil, ch, dunn = np.nan, np.nan, np.nan
                            cm = ClusteringMetric(X=spectrograms_reduced, y_pred=predicted_labels)
                            try:
                                    dunn = cm.dunn_index()
                            except Exception as e:
                                    dunn = np.nan
                            if mask.sum() >= 2 and len(np.unique(labels_for_sil[mask])) >= 2:
                                try:
                                    sil = silhouette_score(spectrograms_reduced[mask], labels_for_sil[mask])
                                    ch = calinski_harabasz_score(spectrograms_reduced[mask], labels_for_sil[mask])
                                
                                except Exception:
                                    sil, ch = np.nan, np.nan  # fa
                            df.at[f"{algorithm}", (reduction, "Seed")].append( seed)
                            df.at[algorithm, (reduction, "Silhouette")].append(
                            f"{(sil):.2f}" if not np.isnan(sil) else "nan"
                        )
                            df.at[algorithm, (reduction, "CH")].append(
                            f"{(ch):.2f}" if not np.isnan(ch) else "nan"
                        )
                            df.at[algorithm, (reduction, "Dunn")].append(
                                f"{(dunn):.2f}" if not np.isnan(ch) else "nan"
                            )
                        else:
                            patience_dict[(algorithm, reduction)] = current_patience + 1

                            if current_patience +1 == 3:
                                print(f"skipping {(algorithm, reduction)}")

    return df



_NUM = re.compile(r"\s*([+-]?\d+(?:\.\d+)?)\s*±\s*([+-]?\d+(?:\.\d+)?)")

def _parse_mean_std(cell):
    if isinstance(cell, (int, float)):
        return float(cell), np.nan
    m = _NUM.match(str(cell))
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2))

def _reduction_family(name: str) -> str:
    n = str(name).lower()
    if n == "none": return "None"
    if n.startswith("t-sne"): return "t-SNE"
    if n.startswith("pca"):   return "PCA"
    if n.startswith("umap"):  return "UMAP"
    return str(name).split("_", 1)[0]

def _algo_family(name: str) -> str:
    n = str(name).lower()
    if n.startswith("kmeans") or n.startswith("k-means"): return "KMeans"
    if n.startswith("agglomerative"): return "Agglomerative"
    if n.startswith("hdbscan"): return "HDBSCAN"
    return str(name).split("_", 1)[0]

def get_mean_std(path):

    df = pd.read_csv(path, header=[0, 1], index_col=0)

    def parse_list(x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return [float(v) for v in x]
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return [float(i) for i in v]
            return [float(v)]
        except Exception:
            s = str(x).strip("[]").replace("'", "")
            try:
                return [float(v) for v in s.split(",") if v.strip()]
            except:
                return []

    df = df.applymap(parse_list)

    def mean_std_str(values):
        if not values:
            return "nan ± nan"
        arr = np.array(values, dtype=float)
        mean, std = arr.mean(), arr.std()
        return f"{mean:.2f} ± {std:.2f}"

    result_df = df.applymap(mean_std_str)
    return result_df

def build_family_score_table_labeled(
    result_df: pd.DataFrame,
    metric: str = "Accuracy",
    family_col_order: tuple[str, ...] = ("None", "t-SNE", "PCA", "UMAP"),
    show_best_in_family_columns: bool = True,
    show_best_in_family_rows: bool = True,
) -> pd.DataFrame:

    if not isinstance(result_df.columns, pd.MultiIndex):
        raise TypeError("result_df must have MultiIndex columns (Reduction, Metric).")

    metric_cols = [c for c in result_df.columns if c[1] == metric]
    if not metric_cols:
        raise ValueError(f"No columns found for metric '{metric}'.")

    red_fam = {c: _reduction_family(c[0]) for c in metric_cols}
    col_fams_all = list({fam for fam in red_fam.values()})
    ordered_col_fams = [f for f in family_col_order if f in col_fams_all] + \
                       sorted([f for f in col_fams_all if f not in family_col_order])

    algo_fams_map = {algo: _algo_family(algo) for algo in result_df.index}
    row_fams_all = list({fam for fam in algo_fams_map.values()})
    row_fams = sorted(row_fams_all, key=lambda x: ["KMeans","Agglomerative","HDBSCAN"].index(x)
                      if x in ["KMeans","Agglomerative","HDBSCAN"] else 9999)

    table = pd.DataFrame(index=row_fams, columns=ordered_col_fams, dtype=object)
    row_family_best_algo = {rf: (None, -np.inf) for rf in row_fams}      
    col_family_best_red  = {cf: (None, -np.inf) for cf in ordered_col_fams}  

    for rf in row_fams:
        algo_rows = [r for r, fam in algo_fams_map.items() if fam == rf]

        if show_best_in_family_rows:
            for r in algo_rows:
                best_mean_for_algo = -np.inf
                for c in metric_cols:
                    m, _ = _parse_mean_std(result_df.at[r, c])
                    if pd.notna(m) and m > best_mean_for_algo:
                        best_mean_for_algo = m
                if best_mean_for_algo > row_family_best_algo[rf][1]:
                    row_family_best_algo[rf] = (r, best_mean_for_algo)

        for cf in ordered_col_fams:
            cols_in_cf = [c for c in metric_cols if red_fam[c] == cf]

            best_mean, best_std = -np.inf, np.nan
            best_algo_name = None
            best_reduct_name = None

            for r in algo_rows:
                for c in cols_in_cf:
                    mean, std = _parse_mean_std(result_df.at[r, c])
                    if pd.isna(mean):
                        continue
                    if mean > best_mean:
                        best_mean, best_std = mean, std
                        best_algo_name = r                 
                        best_reduct_name = c[0]          
                    if show_best_in_family_columns and mean > col_family_best_red[cf][1]:
                        col_family_best_red[cf] = (c[0], mean)

            if np.isfinite(best_mean):
                table.at[rf, cf] = f"{best_mean:.2f} ± {best_std:.2f} ({best_algo_name} | {best_reduct_name})"
            else:
                table.at[rf, cf] = ""

    if show_best_in_family_rows:
        table["Best clustering (name)"] = [row_family_best_algo[rf][0] for rf in row_fams]

    if show_best_in_family_columns:
        bottom = {cf: col_family_best_red[cf][0] for cf in ordered_col_fams}
        if show_best_in_family_rows:
            bottom["Best clustering (name)"] = "" 
        bottom_row = pd.DataFrame([bottom], index=["Best reduction (name)"])
        table = pd.concat([table, bottom_row], axis=0)

    return table


def test_layer (layer, path, algorithm, num_trials = 10):
    num_clusters=[2, 3, 4,5,6,7, 8, 9, 10]
    print(f"processing layer {layer}")
    spectrograms_scaled, true_labels, keys =prepare_data(layer, path)
    reduction_names_umap = get_reduction_names("umap", n_components=[ 2, 10, 50, 100, 200, 300 ],  n_neighbors  =[5, 10, 30, 50, 100, 200], min_dists = [0])
    reduction_names_tsne = get_reduction_names("t-SNE",n_components = [2, 3], perplexities= [  5, 15, 30, 50, 80, 100, 200, 300, 400, 500 ])
    reduction_names_pca = get_reduction_names("PCA",[ 2, 10, 50, 100, 200, 300 ] )
    reduction_names =  reduction_names_tsne  + reduction_names_pca + reduction_names_umap 
    if algorithm == "Agglomerative":
        algorithms_names = get_clustering_names("Agglomerative", linkages = ["ward", "single", "complete", "average"], clusters = num_clusters)
    elif algorithm == "HDBSCAN":
        algorithms_names = get_clustering_names("HDBSCAN", selections = ["eom", "leaf"],  min_samples = [10, 30, 50, 70, 100], min_clusters = [80, 90,  100], clusters = num_clusters)
    elif algorithm == "K-means":
        algorithms_names= get_clustering_names("K-means", clusters = num_clusters)
    save_path =  "/results/raw"
    layer_results_path = os.path.join(save_path, f"layer{layer}")
    os.makedirs(layer_results_path, exist_ok=True)
    print(layer_results_path)
    df =find_best_parameters_supervised (algorithms_names, reduction_names, spectrograms_scaled, num_trials = num_trials, keys = keys)
    
    df_path = os.path.join(layer_results_path, f"{algorithm}_layer{layer}_{num_trials}.csv")
    df.to_csv(df_path)
    mean_std_df = get_mean_std(df_path)
    mean_std_df.to_csv(os.path.join(layer_results_path, f"{algorithm}_layer{layer}_mean_std_{num_trials}.csv"))
    best_results_table = build_family_score_table_labeled(mean_std_df, metric="Silhouette")
    best_results_table.to_csv(os.path.join(layer_results_path, f"{algorithm}_layer{layer}_best_results_{num_trials}_sil.csv"))
    best_results_table = build_family_score_table_labeled(mean_std_df, metric="CH")
    best_results_table.to_csv(os.path.join(layer_results_path, f"{algorithm}_layer{layer}_best_results_{num_trials}_ch.csv"))


if __name__ == "__main__":
    layer = -17
    # example of running for k-means
    test_layer (layer= layer,path = "/beat-this/embeddings", algorithm="K-means", num_trials = 10)