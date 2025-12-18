import uuid
import random
import numpy as np

#====================================================
# SYNNER Functions
#====================================================

C_SYN_UUID = {}

def gen_random_uuid(source_value, column_name):
    if column_name not in C_SYN_UUID:
        C_SYN_UUID[column_name] = {}
    if source_value not in C_SYN_UUID[column_name]:
        C_SYN_UUID[column_name][source_value] = str(uuid.uuid4())
    return C_SYN_UUID[column_name][source_value]
    
def continuous_weighted_average(cluster_neigh_vl, cluster_neigh_ws, decimals=10):
    #print("cluster_neigh_vl",cluster_neigh_vl)
    #print("cluster_neigh_ws",cluster_neigh_ws)
    average = np.average(cluster_neigh_vl,weights=cluster_neigh_ws)
    #print("average",average)
    stddev = np.sqrt(np.cov(cluster_neigh_vl, aweights=cluster_neigh_ws))
    #print("stddev",stddev)
    random_val = random.random()
    #print("random",random_val)
    result = average - stddev + 2*stddev*random_val
    #print("result",result)
    if np.isnan(result):
        return result
    else:
        if decimals == 0:
            try:
                return int(result)
            except ValueError:
                print("Error: value error")
                print("cluster_neigh_vl",cluster_neigh_vl)
                print("cluster_neigh_ws",cluster_neigh_ws)
                print("result",result)
                exit(1)
        else:
            return round(result,decimals)
    

def categorical_weighted_random_choice(cluster_neigh_vl, cluster_neigh_ws):
    cluster_neigh_ws = np.array(cluster_neigh_ws)
    cluster_neigh_ws /= cluster_neigh_ws.sum()
    result_array = np.random.choice(cluster_neigh_vl, 1, p=cluster_neigh_ws)
    return result_array[0]