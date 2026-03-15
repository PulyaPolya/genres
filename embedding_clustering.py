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
# from transformers import ASTConfig
from transformers import ASTModel, ASTConfig
ast_base = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
harmonix_metadata = pd.read_csv("metadata_harmonix.csv")



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

def prepare_data(layer, keys_to_use= None):
    data = load_data(layer)
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
    mean = X.mean(axis=1)          # (N, F)
    std  = X.std(axis=1)           # (N, F)
    p25  = np.percentile(X, 25, axis=1)  # (N, F)
    p75  = np.percentile(X, 75, axis=1)  # (N, F)
    pooled = np.concatenate([mean, std, p25, p75], axis=1)  # (N, 4F)
    return pooled
def prepare_data_raw(layer, path):
    data =np.load(path)
    spectrograms_np, true_labels, keys = prepare_raw_spectrograms(data)
    print(spectrograms_np.shape)
    pooled = pool_time(spectrograms_np)          # (N, 4F)
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


from collections import Counter 
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
