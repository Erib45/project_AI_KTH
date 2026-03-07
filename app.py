from flask import Flask, render_template, request, jsonify
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import os
import random
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- LOAD MODELS & GRAPH (Done once at startup) ---
print("Loading Model, Encoders, and Graph...")
model = load_model("trained_model.keras")

with open("link_encoder.pkl", "rb") as f: link_encoder = pickle.load(f)
with open("target_encoder.pkl", "rb") as f: target_encoder = pickle.load(f)
with open("choice_encoder.pkl", "rb") as f: choice_encoder = pickle.load(f)

graph = nx.read_gml("wikipedia_subset_small.gml")
nodes_list = list(graph.nodes)
# --------------------------------------------------

@app.route('/')
def index():
    # Pass the nodes to the frontend to populate the dropdown menus
    return render_template('index.html', nodes=nodes_list)

@app.route('/play')
def play():
    # Serve the game page
    return render_template('play.html', nodes=nodes_list)

@app.route('/get_neighbors', methods=['POST'])
def get_neighbors():
    # Returns the list of valid links for whatever page the player is currently on
    data = request.json
    node = data.get('node')
    
    if node in graph:
        # Sort them alphabetically so it's easier to read!
        neighbors = sorted(list(graph.neighbors(node)))
        return jsonify({"neighbors": neighbors})
    
    return jsonify({"neighbors": []}), 404

@app.route('/game_setup', methods=['POST'])
def game_setup():
    # Calculates the "Par" score to beat
    data = request.json
    try:
        bfs_path = nx.shortest_path(graph, source=data['start'], target=data['target'])
        return jsonify({"optimal_steps": len(bfs_path) - 1})
    except nx.NetworkXNoPath:
        return jsonify({"optimal_steps": "?"})

@app.route('/get_hint', methods=['POST'])
def get_hint():
    # Finds the correct next step, grabs 2 random wrong steps, and shuffles them
    data = request.json
    node = data['node']
    target = data['target']
    
    neighbors = list(graph.neighbors(node))
    if not neighbors: return jsonify({"hint_options": []})
        
    try:
        path = nx.shortest_path(graph, source=node, target=target)
        correct_next = path[1] if len(path) > 1 else node
    except nx.NetworkXNoPath:
        correct_next = random.choice(neighbors)
        
    other_neighbors = [n for n in neighbors if n != correct_next]
    distractors = random.sample(other_neighbors, min(2, len(other_neighbors)))
    
    hint_options = [correct_next] + distractors
    random.shuffle(hint_options)
    
    return jsonify({"hint_options": hint_options})

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    start = data['start']
    target = data['target']

    # 1. Get True Shortest Path (BFS)
    try:
        bfs_path = nx.shortest_path(graph, source=start, target=target)
    except nx.NetworkXNoPath:
        bfs_path = []

    # 2. Get AI Predicted Path
    node = start
    ai_path = []
    j = 0
    
    while j < 15:
        ai_path.append(node)
        if node == target:
            break
            
        neighbors = list(graph.neighbors(node))
        if not neighbors: # Dead end
            break

        try:
            # THE FIX: No more pandas dataframes. No more replacing commas with spaces!
            # We pass the EXACT comma-separated string the AI trained on.
            links_str = ",".join(neighbors)
            
            links_encoded = link_encoder.transform([links_str])
            target_encoded = target_encoder.transform([target])

            # Predict using our new dual-input Embeddings model
            predictions = model.predict([links_encoded, target_encoded], verbose=0)[0] 
            
            # Constrain the AI to only pick valid, known neighbors
            known_neighbors = [n for n in neighbors if n in choice_encoder.classes_]
            
            if known_neighbors:
                valid_ids = choice_encoder.transform(known_neighbors)
                best_valid_id = max(valid_ids, key=lambda idx: predictions[idx])
                node = choice_encoder.inverse_transform([best_valid_id])[0]
            else:
                # Fallback
                predicted_label = np.argmax(predictions)
                node = choice_encoder.inverse_transform([predicted_label])[0]

        except Exception as e:
            print(f"Encoding Error (unseen node): {e}")
            break
            
        j += 1
        
    if ai_path[-1] != target:
        ai_path.append("(Failed)")

    return jsonify({
        "bfs_path": bfs_path,
        "ai_path": ai_path
    })

if __name__ == '__main__':
    app.run(debug=True)