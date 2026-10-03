import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import os
from scipy.stats import norm
import pulp
import networkx as nx
import glob
import re



flow=pd.read_parquet('flow.parquet')
Sorted_flow_df = flow.sort_values(by='TOA')
Sorted_flow_df =Sorted_flow_df.iloc[:1000,1:]
Link=pd.read_csv("./links.csv")
edges_list=[]
for row_index, (_,_,count1,count2) in Link.iterrows():
    edges = (count1,count2)
    edges_list.append(edges)

# Number of nodes and links
num_nodes = 23
num_links = 74

# Create a graph for the network topology
G = nx.DiGraph()

# Add nodes
for i in range(1,num_nodes):
    G.add_node(i)

# Add edges (example, replace with actual edges)
edges = edges_list

for edge in edges:
    G.add_edge(edge[0], edge[1])

capacity = 50000 #50K

utilization_vector = np.zeros(num_links, dtype= np.float64)


def optimize_flow(Sorted_flow_df, G, edges, capacity, utilization_vector):
    edge_to_index = {edge: idx for idx, edge in enumerate(edges)}

    results = []
    unfeasible_flows = []
    active_flows = []

    # ✅ Track bandwidth usage per edge precisely
    edge_active_flows = {edge: [] for edge in edges}

    for _, row in Sorted_flow_df.iterrows():
        source = row['source_node']
        destination = row['destination_node']
        arrival_time = row['TOA']
        duration = row['Duration']
        bandwidth = row['Bandwidth']

        # print(f"The utilization vector {[f'{x:.2f}' for x in utilization_vector]}")

        active_flows.sort(key=lambda x: x['end_time'])

        # ✅ Remove expired flows and update edge_active_flows
        while active_flows and active_flows[0]['end_time'] <= arrival_time:
            departing_flow = active_flows.pop(0)

            for i in range(len(departing_flow['path']) - 1):
                edge = (departing_flow['path'][i], departing_flow['path'][i + 1])
                if edge in edge_active_flows:
                    # ✅ Remove this specific flow from edge_active_flows
                    edge_active_flows[edge] = [
                        f for f in edge_active_flows[edge]
                        if not (f['start_time'] == departing_flow['start_time'] and f['end_time'] == departing_flow['end_time'])
                    ]

            # ✅ Recompute utilization vector from active edge flows
            for edge in edges:
                total_bw = sum(f['bandwidth'] for f in edge_active_flows[edge])
                utilization_vector[edge_to_index[edge]] = total_bw / capacity

            # ✅ Add departed flow to results
            departed_entry = {
                'source_node': source,
                'destination_node': destination,
                'TOA': departing_flow['start_time'],
                'Duration': departing_flow['end_time'] - departing_flow['start_time'],
                'Bandwidth': - departing_flow['bandwidth'],
                'Path': departing_flow['path']
            }

            
            label = np.zeros(len(edges), dtype=np.int8)
            
            for i in range(len(departing_flow['path'])-1):
                edge = (
                    departing_flow['path'][i],
                    departing_flow['path'][i+1]
                )
                label[edge_to_index[edge]] = -1
            
            for i in range(len(edges)):
                departed_entry[f"Link_{i+1}"] = label[i]


            
            for i in range(len(utilization_vector)):
                departed_entry[f'Utilization_{i+1}'] = utilization_vector[i]

            results.append(departed_entry)

        # Start optimization for incoming flow
        # prob = pulp.LpProblem("Maximize_Link_Utilization", pulp.LpMaximize)
        prob = pulp.LpProblem("RouteFlowFeasibly", pulp.LpMinimize)



        all_paths = list(nx.all_simple_paths(G, source, destination))[:5]

        path_vars = {}
        flow_vars = {}
        for p_idx, path in enumerate(all_paths):
            path_vars[p_idx] = pulp.LpVariable(f"Path_{source}_{destination}_{p_idx}", cat='Binary')
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                if edge not in flow_vars:
                    flow_vars[edge] = pulp.LpVariable(f"Flow_{edge}", lowBound=0, cat='Continuous')


        # Add cost expression to penalize congested and longer paths
        objective_expr = pulp.lpSum(
            path_vars[p_idx] * sum(
                utilization_vector[edge_to_index[(path[i], path[i+1])]] + 0.01
                for i in range(len(path) - 1)
            )
            for p_idx, path in enumerate(all_paths)
        )
        prob += objective_expr




        for edge in edges:
            if edge in flow_vars:
                prob += flow_vars[edge] <= 0.99999 - utilization_vector[edge_to_index[edge]], \
                        f"Capacity_Constraint_{edge}"

        prob += pulp.lpSum(path_vars[p_idx] for p_idx in range(len(all_paths))) == 1, \
                f"SinglePathConstraint_{source}_{destination}"

        for p_idx, path in enumerate(all_paths):
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                if edge in flow_vars:
                    prob += flow_vars[edge] <= path_vars[p_idx] * 1e6, \
                            f"PathConsistency_{source}_{destination}_{p_idx}_{edge}"

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] == "Infeasible":
            unfeasible_flows.append({
                'source_node': source,
                'destination_node': destination,
                'TOA': arrival_time,
                'Duration': duration,
                'Bandwidth': bandwidth,
                **{f'Utilization_{i+1}': utilization_vector[i] for i in range(len(utilization_vector))}})
            continue

        path_taken = None
        for p_idx in range(len(all_paths)):
            if pulp.value(path_vars[p_idx]) > 0:
                path_taken = all_paths[p_idx]
                break

        result_entry = {
            'source_node': source,
            'destination_node': destination,
            'TOA': arrival_time,
            'Duration': duration,
            'Bandwidth': bandwidth,
            'Path': path_taken
        }


        
        label = np.zeros(len(edges), dtype=np.int8)
        
        for i in range(len(path_taken)-1):
            edge = (path_taken[i], path_taken[i+1])
            label[edge_to_index[edge]] = 1
        
        for i in range(len(edges)):
            result_entry[f"Link_{i+1}"] = label[i]


        
        if path_taken:
            # ✅ Add this flow to edge_active_flows
            for i in range(len(path_taken) - 1):
                edge = (path_taken[i], path_taken[i + 1])
                if edge in edge_active_flows:
                    edge_active_flows[edge].append({
                        'start_time': arrival_time,
                        'end_time': arrival_time + duration,
                        'bandwidth': bandwidth
                    })

            # ✅ Recompute utilization vector after adding flow
            for edge in edges:
                total_bw = sum(f['bandwidth'] for f in edge_active_flows[edge])
                utilization_vector[edge_to_index[edge]] = total_bw / capacity

            active_flows.append({
                'start_time': arrival_time,
                'end_time': arrival_time + duration,
                'path': path_taken,
                'bandwidth': bandwidth
            })

            for i in range(len(utilization_vector)):
                result_entry[f'Utilization_{i+1}'] = utilization_vector[i]

            results.append(result_entry)

    return results, unfeasible_flows, utilization_vector




chunk_size = 100
num_chunks = (len(Sorted_flow_df) + chunk_size - 1) // chunk_size

# Initialize storage for results
all_results = []
all_unfeasible_flows = []

utilization_vector = np.zeros(74)

# Main loop: Update to handle utilization vector properly
for i in range(num_chunks):
    flow_chunk = Sorted_flow_df.iloc[i * chunk_size : (i + 1) * chunk_size]

    # print([f"{x:.2f}" for x in utilization_vector])

    # Optimize flow for the current chunk
    results, unfeasible_flows, current_utilization_vector = optimize_flow(
        flow_chunk, G, edges, capacity, utilization_vector
    )
    # print(f"current_utilization_vector {current_utilization_vector}")
    
    utilization_vector = current_utilization_vector.copy()

    # Save intermediate results (feasible flows with utilization vector)
    chunk_MLU_df = pd.DataFrame(results)
    chunk_MLU_df.to_parquet(f'out/MLU_chunk_{i + 1}.parquet', index=False, engine='pyarrow')

    # Save intermediate results for unfeasible flows
    if unfeasible_flows:
        chunk_unfeasible_df = pd.DataFrame(unfeasible_flows)
        chunk_unfeasible_df.to_parquet(f'out/Unfeasible_flows_chunk_{i + 1}.parquet', index=False, engine='pyarrow')

    # ✅ Save first 10,000 results for preview (from first chunk)
    if i == 0:
        preview_df = pd.DataFrame(results[:10000])
        preview_df.to_csv('out/Preview_First_10000.csv', index=False)

    print(f"Chunk {i + 1}/{num_chunks} processed.")




# # # Get all chunk files
# chunk_files = glob.glob("out/Unfeasible_flows_chunk_*.parquet")
chunk_files = glob.glob("out/MLU_chunk_*.parquet")

# # # Sort files based on numerical order (extracting the chunk number)
# chunk_files.sort(key=lambda x: int(re.search(r"Unfeasible_flows_chunk_(\d+)", x).group(1)))
chunk_files.sort(key=lambda x: int(re.search(r"MLU_chunk_(\d+)", x).group(1)))

# # # Read and concatenate all chunk files
df_list = [pd.read_parquet(file) for file in chunk_files]
final_df = pd.concat(df_list, ignore_index=True)

final_df.drop(columns=["Path","TOA","Duration"],axis=1,inplace=True)

# # # Select columns to one-hot encode
columns_to_encode = ["source_node", "destination_node"]

# # # Perform one-hot encoding
final_df = pd.get_dummies(final_df, columns=columns_to_encode, dtype=int)

# # # Move the first column to the last position
first_column = final_df.columns[0]
final_df = final_df[[col for col in final_df.columns if col != first_column] + [first_column]]

final_df.to_parquet("Final_MLU.parquet", index=False)

final_df[0:100].to_csv(
    "Final_MLU.csv",
    index=False)
print("Final merged MLU file saved as out/Final_MLU.feather")
print(final_df.shape)
