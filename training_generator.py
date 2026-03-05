import random
import networkx as nx
import csv
import time

def generate_data(graph, output_file, line_limit):
    print("Optimizing: Precomputing node neighbors...")
    # OPTIMIZATION 1: Pre-join neighbors. 
    # Doing string joins 1,000,000 times in a loop is a massive CPU drain.
    neighbors_map = {node: ','.join(list(graph.neighbors(node))) for node in graph.nodes}
    nodes = list(graph.nodes)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["links", "target", "best_choice"])
        writer.writeheader()
        
        lines_written = 0
        print(f"Generating up to {line_limit} lines of training data...")
        start_time = time.time()
        
        while lines_written < line_limit:
            start_node = random.choice(nodes)
            
            # OPTIMIZATION 2: Bulk Pathfinding
            # Instead of finding a path to ONE target, this gets the shortest paths 
            # to ALL reachable targets in the entire graph in a single sweep.
            paths_from_start = nx.shortest_path(graph, source=start_node)
            
            targets = list(paths_from_start.keys())
            if start_node in targets:
                targets.remove(start_node) # Don't path to itself
            
            if not targets:
                continue
                
            # Pick a sample of targets from this tree (e.g., 50) so we don't 
            # over-represent a single start node in our dataset
            sampled_targets = random.sample(targets, min(50, len(targets)))
            
            for target in sampled_targets:
                path = paths_from_start[target]
                
                for i in range(len(path) - 1):
                    writer.writerow({
                        "links": neighbors_map[path[i]],
                        "target": target,
                        "best_choice": path[i + 1]
                    })
                    lines_written += 1
                    
                    if lines_written >= line_limit:
                        break
                if lines_written >= line_limit:
                    break

        # OPTIMIZATION 3: Coverage Guarantee
        # Your original code wisely ensured every node was a source/target at least once.
        print("Ensuring 100% node coverage for the neural network...")
        
        for node in nodes:
            # 1. Node as a Source
            try:
                target = random.choice(nodes)
                path = nx.shortest_path(graph, source=node, target=target)
                for i in range(len(path) - 1):
                    writer.writerow({
                        "links": neighbors_map[path[i]],
                        "target": target,
                        "best_choice": path[i + 1]
                    })
            except nx.NetworkXNoPath: pass

            # 2. Node as a Target
            try:
                source = random.choice(nodes)
                path = nx.shortest_path(graph, source=source, target=node)
                for i in range(len(path) - 1):
                    writer.writerow({
                        "links": neighbors_map[path[i]],
                        "target": node,
                        "best_choice": path[i + 1]
                    })
            except nx.NetworkXNoPath: pass

        end_time = time.time()
        print(f"Complete! Generated data in {end_time - start_time:.2f} seconds.")

# create a directed graph
print("Loading graph... \n")
graph = nx.read_gml("wikipedia_subset_small.gml")
print("Generating data... ")
generate_data(graph, "output_data.csv", 4_000_000)