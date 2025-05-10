from itertools import product
import os
import numpy as np
import pandas as pd
import subprocess

# utils and src from ABCD_generator.jl
julia = "/home/quak/julia-1.10.8/bin/julia"

n_s = [100_000]
o_s = [5000]
eps_s = [0.7, 0.3]
# o_s = [i for i in [10, 30, 50, 70,100, 200]]
# eps_s = [i/10 for i in [1, 3, 5, 7]]


for n, o, eps in product(n_s, o_s, eps_s):
    folder_name = 'abcdo_data'
    folder_name += f"_{n}_{o}_{eps}"
    print(folder_name)
    os.makedirs(folder_name, exist_ok=True)
    
    subprocess.run(f"{julia} ../utils/deg_sampler.jl\
                   {folder_name}/degrees.dat\
                   2.5 5 25 {n} 10000 42 {o}", shell=True, stdout=subprocess.PIPE)
                    # τ₁ d_min d_max n max_iter
    subprocess.run(f"{julia} ../utils/com_sampler.jl\
                   {folder_name}/community_sizes.dat\
                   1.5 2000 15000 {n} 10000 42 {o}", shell=True, stdout=subprocess.PIPE)
                    # τ₂ c_min c_max n max_iter
    subprocess.run(f"{julia} ../utils/graph_sampler.jl\
                   {folder_name}/edge.dat\
                   {folder_name}/com.dat\
                   {folder_name}/degrees.dat\
                   {folder_name}/community_sizes.dat\
                   xi {eps} false false 42 {o}", shell=True, stdout=subprocess.PIPE)
                    # mu|xi fraction isCL islocal
    # os.chdir('../outliers_small')
    edge_file_path = f"{folder_name}/edges.csv"
    node_graph_mapping_file_path = f"{folder_name}/graph_mapping.csv"
    features_file_path = f"{folder_name}/features.csv"

    np.random.seed(10)

    # create edge file
    edge_file = pd.read_csv(f"{folder_name}/edge.dat", delimiter="\t", header=None)
    edge_file.columns = ["src_node_id", "dest_node_id"]
    edge_file["src_node_id"] -=1
    edge_file["dest_node_id"] -=1

    communities = pd.read_csv(f"{folder_name}/com.dat", delimiter="\t", header=None)
    communities.columns = ["node_id", "community_id"]
    communities["node_id"] -= 1
    communities["community_id"] -= 1


    features = []
    for community in list(set(communities["community_id"])):
        mu, std = np.random.normal(0, 30), np.random.exponential(10)
        comm_df = communities.query(f"community_id=={community}")
        size = len(comm_df)
        random_community_feature = np.random.normal(mu, std, size)
        is_outlier = 1 if community == 0 else 0
        features += [
            (i, v, c, is_outlier)
            for i, v, c in zip(
                comm_df["node_id"].values,
                random_community_feature,
                comm_df["community_id"].values,
            )
        ]

    features = pd.DataFrame(
        features, columns=["node_id", "random_community_feature", "community_id", 'is_outlier']
    )


    node_graph_mapping_file = communities.copy().drop(columns=["community_id"])
    node_graph_mapping_file["graph_id"] = 0

    edge_file.to_csv(edge_file_path, index=False)
    node_graph_mapping_file.to_csv(node_graph_mapping_file_path, index=False)
    # communities.to_csv(communities_file_path, index=sFalse)
    features.to_csv(features_file_path, index=False)
    