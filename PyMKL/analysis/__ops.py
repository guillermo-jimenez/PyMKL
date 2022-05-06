from typing import Union, List, Tuple, Callable

import itertools
import math
import datetime
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import var
import scipy as sp
import scipy.spatial
import scipy.spatial.distance
import numpy.matlib
import tqdm
import sklearn
import sklearn.cluster
import networkx as nx

def INEXACT_Bermanis_4DATA(Ge_s, f, embedding, direction, e_s, gamma):
    # Get shape
    N = Ge_s.shape[0]

    # Solve system
    c = np.linalg.solve((Ge_s + (np.eye(N,N)/gamma) ), f.T)

    # ???
    f_s = Ge_s @ c;

    # step 5
    p = direction.shape[0]
    l = f_s.shape[0]
    f_star_s = np.zeros((f_s.shape[1],p))

    for j in range(p):
        tmp = np.matlib.repmat(direction[j,:], l ,1)
        tmp = np.sum((tmp - embedding)**2,1).T
        G_star_s = np.exp( -tmp / (2*(e_s)**2) ) 
        
        # step 6
        f_star_s[:,j] = (G_star_s @ c).T

    output = {
        "f_s": f_s.T,
        "f_star_s": f_star_s
    }

    return output


def get_sample(Features: List[Union[np.ndarray, list]], index: int):
    point_features = [f[:,index] for f in Features]

    return point_features

def get_variability_descriptors(embedding: np.ndarray, Features: List[Union[np.ndarray, list]], dimensions: Union[None,int,list] = None, direction: np.ndarray = None, gamma: Union[float,np.ndarray] = None, max_iterations: int = 20, NN_dims: Union[int,None] = None):
    """
    Arguments:
    * Features: List[Union[np.ndarray, list]]
        Original input features

    * embedding: np.ndarray
        Data projected into the output space

    * dimensions: Union[int,list]
        Dimensions to be regressed

    * direction: np.ndarray
        Direction of the projection into the output space
    
    * gamma: array_like
        Regularization term

    * max_iterations: int
        Maximum number of iterations for the approximation

    * NN_dims: int
        Number of dimensions to consider when computing the distances for the nearest neighbours to each point

    """
    # Dimensions to array
    if dimensions is None:
        dimensions = embedding.shape[1]
    if isinstance(dimensions,(int,np.integer)):
        dimensions = np.arange(dimensions)
    dimensions = np.array(dimensions)

    # Get subset of interesting dimensions
    embedding_reduced = embedding[:,:NN_dims]

    # Populate optional parameters
    if gamma is None:
        gamma = 1
    if isinstance(gamma,(int,float,np.integer,np.floating)):
        gamma = np.full((len(Features),),gamma)
    gamma = np.array(gamma)

    # Get distance of every projected element to each other
    embedding_distance = sp.spatial.distance.squareform(sp.spatial.distance.pdist(embedding_reduced)) 
    embedding_distance_density = embedding_distance + np.diag(np.full((embedding_reduced.shape[0],),np.inf)) ## put diagonal coefficients to -1
    embedding_distance_density = np.min(embedding_distance_density, 0) ## find closest neighbour
    diameter = np.max(embedding_distance)
    density = np.mean(embedding_distance_density)

    # Iterator depending if it's list (first N dimensions) or list (specific dimensions)
    if isinstance(dimensions,np.integer):
        dimensions = np.arange(dimensions)

    # MSE Algorithm - same in every iteration, take out of loop for efficiency
    if direction is None:
        embedding_mean = np.mean(embedding_reduced,axis=0)
        direction = np.zeros((len(dimensions)*5,embedding_reduced.shape[1]))

        for i,d in enumerate(dimensions):
            std = np.std(embedding_reduced[:,d],ddof=1)
            variability = np.array([-2*std,-std,0,std,2*std])

            for j,var in enumerate(variability):
                direction[5*i+j,:] = embedding_mean
                direction[5*i+j,i] = direction[5*i+j,i] + var
    else:
        # If provided, take NN_dims dimensions
        direction = direction.copy()[:,:NN_dims]

    # Initialize outputs
    outputs = []

    # Inexact bermanis? Dig into function
    for n,f in enumerate(tqdm.tqdm(Features)):
        s = 0
        F_s_old = np.zeros_like(f)
        F_star_s_old = np.zeros((f.shape[0],direction.shape[0]))
        while (s <= max_iterations ) & ((diameter/2**s) > (2*density)):
            # Kernel bandwidth
            e_s = diameter/2**s
            Ge_s = np.exp( -embedding_distance**2 / (2*(e_s)**2) );

            # Compute inexact Bermanis
            bermanis = INEXACT_Bermanis_4DATA(Ge_s, f-F_s_old, embedding_reduced, direction, e_s, gamma[n])

            # Update f_s
            F_s = F_s_old + bermanis["f_s"]
            F_s_old = F_s # Store last value

            # Update f_star_s
            F_star_s = F_star_s_old + bermanis["f_star_s"]
            F_star_s_old = F_star_s # Store last value

            # Increment counter
            s = s+1

        F_star = F_star_s_old

        # Store outputs
        outputs.append(F_star)

    return outputs


def regression_line_clusters(embedding: np.ndarray, clusters: Union[int,np.ndarray], dim2sort: int = None, dimensions: Union[None, int, list] = None, random_state: Tuple[int,np.random.RandomState] = None):
    # Treat inputs - dimensions
    if dimensions is None:
        dimensions = embedding.shape[1]
    if isinstance(dimensions,(int,np.integer)):
        dimensions = np.arange(dimensions)
    dimensions = np.array(dimensions)
    # Treat inputs - clusters
    if isinstance(clusters,(int,np.integer)):
        clusters = np.arange(clusters)
        needs_training = True
    else:
        if len(clusters) == embedding.shape[0]:
            labels = clusters # Keep same variable names
            needs_training = False
        else:
            raise ValueError("clusters can only be either the label for each sample or the number of clusters to partition the data in")

    # Auxiliary list of classes for later
    classes = np.unique(clusters)
    
    # Copy embedding (just in case)
    embedding = np.copy(embedding)
        
    # Clusterize data
    direction = np.full((len(classes),len(dimensions)),np.nan)
    if needs_training:
        has_clustered = False
        while not has_clustered:
            index_to_delete = []
            
            # Clusterize data to 'clusters' clusters
            kmeans = sklearn.cluster.KMeans(n_clusters=clusters, n_init=50, random_state=random_state)
            labels = kmeans.fit_predict(embedding[:,dimensions])
            
            for i,c in enumerate(classes):
                condition = (labels == c)
                if np.sum(condition) < 2:
                    index_to_delete.append(np.where(condition)[0])
            
            if len(index_to_delete) == 0:
                has_clustered = True
            else:
                # Sort from largest to smallest index
                index_to_delete = np.concatenate(index_to_delete) # Form a single array
                index_to_delete = np.sort(index_to_delete)[::-1]
                
                for index in index_to_delete:
                    embedding = np.delete(embedding,index,axis=0)
    
    # Fill dimensions
    for i,c in enumerate(classes):
        direction[i,:] = np.mean(embedding[labels == c,:][:,dimensions],axis=0)
        
    if dim2sort is not None:
        direction = direction[np.argsort(direction[:,dim2sort]),:]
    
    return direction
        
    
def regression_line_points(embedding: np.ndarray, point_from: np.ndarray, point_to: np.ndarray, 
                           n_points: int = 10, dimensions: Union[None,int, list] = None, 
                           metric: Callable = np.median):
    # Treat inputs - dimensions
    if dimensions is None:
        dimensions = embedding.shape[1]
    if isinstance(dimensions,(int,np.integer)):
        dimensions = np.arange(dimensions)
    dimensions = np.array(dimensions)
    
    # Fill points with mean/median data according to metric
    if point_from.size < dimensions.size:
        point_from = np.concatenate((point_from,metric(embedding[:,point_from.size:],axis=0)))
    if point_to.size   < dimensions.size:
        point_to   = np.concatenate((point_to,  metric(embedding[:,point_to.size:  ],axis=0)))
        
    # Determine the empty direction vector
    direction = (point_to - point_from)/(n_points-1)
    points = np.array([point_from+i*direction for i in range(n_points)])

    return points

def regression_line_dimensions(embedding: np.ndarray, dimension_to_explore: int, n_points: int = 10, dimensions: Union[None,int, list] = None, metric: Callable = np.median):
    # Treat inputs - dimensions
    if dimensions is None:
        dimensions = embedding.shape[1]
    if isinstance(dimensions,(int,np.integer)):
        dimensions = np.arange(dimensions)
    dimensions = np.array(dimensions)
    
    # Determine the empty direction vector
    direction = np.full((n_points,len(dimensions)),np.nan)
    embedding_reduced = embedding[:,dimension_to_explore]
    boundaries = np.linspace(np.min(embedding_reduced),np.max(embedding_reduced),n_points+1)

    for i,n in enumerate(range(n_points)):
        condition_boundaries = (embedding_reduced>boundaries[i]) & (embedding_reduced<boundaries[i+1])
        direction[i,:] = metric(embedding[condition_boundaries,:][:,dimensions],axis=0)

    return direction


def regression_line_std(embedding: np.ndarray, n_points: int = 5, dimensions: Union[None,int, list] = None):
    # Treat inputs - dimensions
    if dimensions is None:
        dimensions = embedding.shape[1]
    if isinstance(dimensions,(int,np.integer)):
        dimensions = np.arange(dimensions)
    dimensions = np.array(dimensions)
    
    # MSE Algorithm - same in every iteration, take out of loop for efficiency
    embedding_mean = np.mean(embedding,axis=0)
    direction = np.matlib.repmat(embedding_mean,len(dimensions)*n_points,1)
    for i,d in enumerate(dimensions):
        std = np.std(embedding[:,d],ddof=1)
        variability = np.linspace(-2*std,2*std,n_points)

        for j,var in enumerate(variability):
            direction[n_points*i+j,i] = direction[n_points*i+j,i] + var
        
    return direction



def embedding_self_correlation(embedding: np.ndarray, display_dimensions: bool = True, correlation_threshold: float = 0.9, metric: Callable = lambda x: scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(x,metric="euclidean"))) -> Tuple[np.ndarray,np.ndarray]:
    """This code is used to assess the number of relevant dimensions to
    consider from an MKL output space. To do so, it ranks the neighbors for
    each of the data entry (patient) and compares it to the same ranking but
    computed with one more dimension. This way, when the ranking of nearest
    neighbors does not change, we can argue that the space has stabilized.
    
    Inputs:
    * embedding: embedding of the input data in the latent space"""
    
    # Retrieve input shape
    N,M = embedding.shape
    M_min = M # Initialize to have the same amount of dimensions to analyze

    self_correlation = np.zeros((M,))

    # Iterator to display progress
    try:
        import tqdm
        iterator = tqdm.tqdm(range(1,M+1))
    except ModuleNotFoundError:
        iterator = range(1,M+1)

    for dim in iterator:
        if dim == 1: # Keep tqdm with M iterations for easier understanding
            continue
            
        # ???
        rank_vectors = np.zeros((N,))

        # Compute euclidean distance matrices of 'dim' and 'dim+1' dimensions
        k1 = metric(embedding[:,:dim-1])
        k2 = metric(embedding[:,:dim])

        # Sort pairwise distances
        argsort_k1 = np.argsort(k1)
        argsort_k2 = np.argsort(np.argsort(k2))

        # Save up time
        arange = np.arange(N)

        # Compute rankings for the different number of dimensions
        for aux_dim in range(N):
            # Take index in the value-sorted embedding
            sort_k1 = argsort_k1[aux_dim,:]
            sort_k2 = argsort_k2[aux_dim,:]

            # Sort k2 w.r.t. k1
            sort_k2 = sort_k2[sort_k1]

            # Order based Spearman/Kendall rank correlation
            rank_vectors[aux_dim] = sp.stats.spearmanr(arange,sort_k2)[0]

        # Set minimum number of dimensions when correlation is over 'correlation_threshold'
        self_correlation[dim-1] = np.mean(rank_vectors) # Average the correlation of all patient rankings
        if (self_correlation[dim-1] >= correlation_threshold):
            M_min = dim-1
            break

    return self_correlation,M_min


def cluster_MKR(embedding: np.ndarray, Features: List[np.ndarray], clusters: np.ndarray, dimensions: Union[None,int,np.ndarray] = None, NN_dims: Union[int,None] = None, return_embeddings: bool = False):
    """Regression on cluster modes"""
    # Treat inputs - dimensions
    if dimensions is None:
        dimensions = embedding.shape[1]
    if isinstance(dimensions,(int,np.integer)):
        dimensions = np.arange(dimensions)
    dimensions = np.array(dimensions)

    # Output structures
    descriptors = []
    out_embeddings = []

    # Obtain the per-cluster main directions and descriptors
    for i,c in enumerate(np.unique(clusters)):
        # Retrieve samples that belong to specific cluster
        filter_cluster = (clusters == c)

        # Take the elements corresponding to a certain cluster and center at zero
        embedding_cluster = embedding[filter_cluster,:]
        embedding_cluster -= np.matlib.repmat(np.mean(embedding_cluster,axis=0),embedding_cluster.shape[0],1)

        # Compute PCA
        _,V = np.linalg.eigh(np.cov(embedding_cluster.T),UPLO="U")
        V = np.fliplr(V).T
        embedding_PCA = (V @ embedding_cluster.T).T

        # Get features per cluster
        Features_cluster = [f[:,filter_cluster] for f in Features]

        # Get variability descriptors
        d = get_variability_descriptors(embedding_PCA,Features_cluster,dimensions=dimensions,NN_dims=NN_dims)

        # Save outputs
        descriptors.append(d)
        out_embeddings.append(embedding_PCA)

    if return_embeddings:
        return descriptors,out_embeddings
    else:
        return descriptors


def compute_agreement(input, target):
    """Function for calculating clustering accuray and matching found 
    labels with true labels. Assumes input and target both are Nx1 vectors with
    clustering labels. Does not support fuzzy clustering.
    
    Algorithm is based on trying out all reorderings of cluster labels, 
    e.g. if input = [1 2 2], it tries [1 2 2] and [2 1 1] so see which fits
    best the truth vector. Since this approach makes use of perms(),
    the code will not run for unique(input) greater than 10, and it will slow
    down significantly for number of clusters greater than 7.
    
    Input:
      input  - result from clustering (y-test)
      target - truth vector
    
    Output:
      accuracy    -   Overall accuracy for entire clustering (OA). For
                      overall error, use OE = 1 - OA.
      true_labels -   Vector giving the label rearangement witch best 
                      match the truth vector (target).
      CM          -   Confusion matrix. If unique(input) = 4, produce a
                      4x4 matrix of the number of different errors and  
                      correct clusterings done."""
    
    N = target.size

    # Get the number of clusters
    clusters = np.unique(input)

    # Get all possible permutations of available cluster IDs
    permutations = np.vstack(list(itertools.permutations(clusters)))[::-1]

    # Check which permutation yields the closest response to any given cluster
    agreements = []
    for i,permutation in enumerate(permutations):
        # Reassign the labels to a certain permutation of elements
        flipped_labels = permutation[np.argmax((input == permutation[:,None])[::-1],axis=0)]

        # Save value
        agreements.append(np.sum(flipped_labels == target)/N)

    return agreements,np.max(agreements)


def process_table(data: pd.DataFrame, clusters: np.ndarray, y: Union[str,np.ndarray,list] = "outcome", dtypes: dict = {}):
    # Check inputs
    if isinstance(y, str):
        if y not in data:
            raise KeyError("if y is a string, the Dataframe object *must* contain said key")
    else:
        if data.shape[0] != len(y):
            raise ValueError("if y is an array, the length of the vector must match the number of samples")
    
    # Copy data just in case
    data = data.copy()
    dtypes = dtypes.copy()

    # Get variable types for checking statistical test to be used
    var_types = {}

    # Skip these dtypes
    skip_dtypes = (datetime.datetime, np.datetime64)

    for k in data:
        if k in dtypes:
            continue
        
        mark_break = False
        for v in data[k]:
            if isinstance(v,skip_dtypes):
                mark_break = True

        if mark_break:
            var_types[k] = None
            continue
    
        # Cast if dtype is object
        if data[k].dtype == np.object_:
            try:
                data[k] = pd.to_numeric(data[k])
            except ValueError:
                var_types[k] = None
                continue

        values = data[k][~np.isnan(data[k])]
        unique_values = np.unique(values)

        # Check if values are categorical (no decimal numbers)
        if np.allclose(unique_values,unique_values.astype(int).astype(values.dtype)):
            if unique_values.size == 2:
                var_types[k] = bool
            else:
                var_types[k] = int
        else:
            var_types[k] = float
        
    # Get filters for every cluster
    filters_cluster = {c: (clusters == c) for c in np.unique(clusters)}
    
    # Iterate over variables
    outputs = {}#[None for _ in range(len(data_dict.keys()))]
    for i,k in enumerate(data):
        # Skip if not castable
        if var_types[k] is None:
            continue

        # Add space for information
        outputs[k] = {}

        # Get non-NaN values
        if var_types[k] == bool:
            non_nan_data = data[k][~np.isnan(data[k])].values.astype(float)
            non_nan_data = (non_nan_data-non_nan_data.min()).astype(var_types[k])
        else:
            non_nan_data = data[k][~np.isnan(data[k])].values.astype(var_types[k])
        non_nan_clusters = clusters[~np.isnan(data[k])]
        if (non_nan_data.size == 0) or (non_nan_clusters.size == 0):
            outputs[k]["p-value"] = np.NaN

            for c in np.unique(clusters):
                outputs[k][f"Cluster {c} (n = {np.sum(filters_cluster[c])})"] = f"---"
            continue
        is_normal = sp.stats.kstest(sp.stats.zscore(non_nan_data),"norm").pvalue >= 5e-2
        
        # Check ranges
        groups = []
        for c in np.unique(clusters):
            # Get elements that coincide with the cluster
            cluster_data         = data[k].values[filters_cluster[c]]
            if var_types[k] == bool:
                non_nan_cluster_data = cluster_data[~np.isnan(cluster_data)].astype(float)
                if (np.unique(data[k].values).size == 2):
                    non_nan_cluster_data = (non_nan_cluster_data-np.unique(data[k].values).min()).astype(var_types[k])
            else:
                non_nan_cluster_data = cluster_data[~np.isnan(cluster_data)].astype(var_types[k])

            if (cluster_data.size == 0) or (non_nan_cluster_data.size == 0):
                outputs[k][f"Cluster {c} (n = {np.sum(filters_cluster[c])})"] = f"---"
                continue
            
            # Get groups of data
            groups.append(non_nan_cluster_data)
            
            if var_types[k] is bool: # Case binary
                to_bool      = (non_nan_cluster_data == True)
                outputs[k][f"Cluster {c} (n = {np.sum(filters_cluster[c])})"] = f"{to_bool.sum()} ({np.round(100*np.mean(to_bool),2)})%"
            elif is_normal:
                mean         = np.mean(non_nan_cluster_data)
                std          = np.std(non_nan_cluster_data,ddof=1)
                outputs[k][f"Cluster {c} (n = {np.sum(filters_cluster[c])})"] = f"{np.round(mean,2)} ± {np.round(std,2)}"
            else:
                median       = np.median(non_nan_cluster_data)
                y_25         = np.quantile(non_nan_cluster_data,0.25, interpolation='midpoint')
                y_75         = np.quantile(non_nan_cluster_data,0.75, interpolation='midpoint')
                outputs[k][f"Cluster {c} (n = {np.sum(filters_cluster[c])})"] = f"{np.round(median,2)} ({np.round(y_25,2)} to {np.round(y_75,2)})"
    
        # Compute p-values
        if (non_nan_data.size == 0) or (non_nan_clusters.size == 0):
            outputs[k]["p-value"] = np.NaN
            continue

        if (var_types[k] is bool) or (k == y):
            # Compute chi-square test for statistical significance among groups
            crosstab = pd.crosstab(data[k].values,clusters).values
            _,outputs[k]["p-value"],_,_ = sp.stats.chi2_contingency(crosstab, correction=False)
        else:
            if (len(groups) < 2):
                outputs[k]["p-value"] = np.NaN
                continue
            elif is_normal: # If is normal, Kruskal-Wallis test
                _,outputs[k]["p-value"] = sp.stats.kruskal(*groups)
            else:
                _,outputs[k]["p-value"] = sp.stats.f_oneway(*groups)

        # if k == y:
        #     return non_nan_data,non_nan_clusters
            
    return pd.DataFrame(outputs).T

def get_graph(embedding: np.ndarray, N_neighbours: int = 5, NN_dims: Union[int,None] = None, labels: list = None):
    """
    Arguments:
    * Features: List[Union[np.ndarray, list]]
        Original input features

    * embedding: np.ndarray
        Data projected into the output space

    * dimensions: Union[int,list]
        Dimensions to be regressed

    * direction: np.ndarray
        Direction of the projection into the output space
    
    * gamma: array_like
        Regularization term

    * max_iterations: int
        Maximum number of iterations for the approximation

    * NN_dims: int
        Number of dimensions to consider when computing the distances for the nearest neighbours to each point

    """
    # Dimensions to array
    if NN_dims is None:
        NN_dims = embedding.shape[1]
    if isinstance(NN_dims,(int,np.integer)):
        NN_dims = np.arange(NN_dims)
    NN_dims = np.array(NN_dims)

    # Get subset of interesting dimensions
    embedding_reduced = embedding[:,NN_dims]

    # Get distance of every projected element to each other
    embedding_distance  = sp.spatial.distance.squareform(sp.spatial.distance.pdist(embedding_reduced)) 
    embedding_distance += np.diag(np.full((embedding_reduced.shape[0],),np.inf)) ## put diagonal coefficients to -1
     
    # Find N neighbours
    neighbours          = np.argsort(embedding_distance,axis=1)[:,:N_neighbours]
    
    # Create graph
    g = nx.Graph()
    for i,nn in enumerate(neighbours):
        for j in nn:
            g.add_edge(i,j,weight=embedding_distance[i,j])
    if labels is not None:
        nx.set_node_attributes(g,{i: {"label": v} for i,v in enumerate(labels)})
    
    return g