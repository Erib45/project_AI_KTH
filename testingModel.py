import random
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model

link_encoder = LabelEncoder()
target_encoder = LabelEncoder()
choice_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder()
model = load_model("trained_model.keras")

# Load the graph
print("Loading graph... \n")
graph = nx.read_gml("wikipedia_subset_small.gml")

def find_shortest_path(graph, start, target):
    # Find the shortest path between two nodes in the graph using networkx built in function
    try:
        return nx.shortest_path(graph, source=start, target=target)
    except nx.NetworkXNoPath:
        return []

start, target = random.sample(list(graph.nodes), 2)
neighbors = list(graph.neighbors(start))
new_data = pd.DataFrame({
    'links': [",".join(neighbors)],
    'target': [target]
})
new_data['links'] = new_data['links'].apply(lambda x: x.split(','))  # Split the links
new_data['links'] = new_data['links'].apply(lambda x: ' '.join(x))  # Join as a single string
new_data['links'] = link_encoder.fit_transform(new_data['links'])
new_data['target'] = target_encoder.fit_transform(new_data['target'])
i = 0
while i < 10:
    start, target = random.sample(list(graph.nodes), 2)
    print(f"start: '{start}'\ntarget: '{target}'")

    path = find_shortest_path(graph, start, target)
    if(path == []):
        continue
    print(f"best path:\t{path}")

    print("ai path: ")
    node = start
    j = 0
    AIpath = []
    while j < 10:
        AIpath.append(node)
        neighbors = list(graph.neighbors(node))
        new_data = pd.DataFrame({
            'links': [",".join(neighbors)],
            'target': [target]
        })

        # Encode the new data
        new_data['links'] = new_data['links'].apply(lambda x: x.split(','))
        new_data['links'] = new_data['links'].apply(lambda x: ' '.join(x))
        new_data['links'] = link_encoder.transform(new_data['links'])
        new_data['target'] = target_encoder.transform(new_data['target'])

        # One hot encode the new data
        new_data_encoded = one_hot_encoder.transform(new_data).toarray()

        predictions = model.predict(new_data_encoded, verbose=0)
        predicted_label = np.argmax(predictions, axis=1)
        predicted_best_choice = choice_encoder.inverse_transform(predicted_label)

        node = predicted_best_choice[0]
        if node == target:
            break
        j += 1
    AIpath.append(node)
    print(AIpath)
    print("\n\n")
    i += 1