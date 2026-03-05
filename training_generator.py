import random
import networkx as nx
import csv

def find_shortest_path(graph, start, target):
    #Find the shortest path between two nodes in the graph using networkx built in function
    try:
        return nx.shortest_path(graph, source=start, target=target)
    except nx.NetworkXNoPath:
        return []

def generate_data(graph, output_file, line_limit):
    #Open the output file
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["links", "target", "best_choice"])
        writer.writeheader()
        
        lines_written = 0
        #While the lines written is less than the line limit
        while lines_written < line_limit:
            #Randomly take two nodes from the graph
            start, target = random.sample(list(graph.nodes), 2)

            #Find the shortest path between the two nodes
            path = find_shortest_path(graph, start, target)
            
            #If no path is found, skip to the next iteration
            if not path:
                continue  

            #Loop through the path and for each node, write the node, its neighbors, the target node, and the best choice to the output file
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]

                row = {
                    "links": ','.join(list(graph.neighbors(current_node))),
                    "target": target,
                    "best_choice": next_node
                }

                writer.writerow(row)
                lines_written += 1

                if lines_written >= line_limit:
                    break
        for node in graph.nodes:
            target = random.sample(list(graph.nodes), 1)[0]
            path = find_shortest_path(graph, node, target)
            for i in range(len(path) - 1):
                currentNode = path[i]
                nextNode = path[i + 1]
                row = {
                    "links": ','.join(list(graph.neighbors(currentNode))),
                    "target": target,
                    "best_choice": nextNode
                }
                writer.writerow(row)
        for node in graph.nodes:
            source = random.sample(list(graph.nodes), 1)[0]
            path = find_shortest_path(graph, source, node)
            for i in range(len(path) - 1):
                currentNode = path[i]
                nextNode = path[i + 1]
                row = {
                    "links": ','.join(list(graph.neighbors(currentNode))),
                    "target": node,
                    "best_choice": nextNode
                }
                writer.writerow(row)
# create a directed graph
print("Loading graph... \n")
graph = nx.read_gml("wikipedia_subset_small.gml")
print("Generating data... ")
generate_data(graph, "output_data.csv", 1_000_000)
print("Complete!")