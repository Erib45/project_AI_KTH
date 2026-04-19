import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request, jsonify
import networkx as nx
import pandas as pd
from sb3_contrib import MaskablePPO
import numpy as np
import pickle
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)
import random
import time
from collections import deque
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

print("Loading Reinforcement Learning Agent...")
try:
    rl_model = MaskablePPO.load("rl_wiki_model")
    # The RL AI needs to translate string names into integer IDs
    node_to_id = {node: i for i, node in enumerate(nodes_list)}
    id_to_node = {i: node for i, node in enumerate(nodes_list)}
    print("RL Agent Ready!")
except Exception as e:
    print(f"Warning: Could not load RL model. Did you run the trainer script? {e}")
    rl_model = None

# 🚨 THE FIX: WARM UP THE NEURAL NETWORK 🚨
print("Warming up the AI engine...")
# We feed it a fake, random prediction so TensorFlow builds its graph now, not during the game.
dummy_neighbors = ",".join(list(graph.neighbors(nodes_list[0])))
dummy_link = link_encoder.transform([dummy_neighbors])
dummy_target = target_encoder.transform([nodes_list[1]])
model([dummy_link, dummy_target], training=False)
print("System Ready!")
# --------------------------------------------------
def custom_bfs(graph, start, target):
    if start not in graph or target not in graph:
        return [], 0
    if start == target:
        return [start], 1
    
    queue = deque([[start]])
    visited = {start}
    nodes_explored = 0
    
    while queue:
        path = queue.popleft()
        current_node = path[-1]
        nodes_explored += 1 # Count every node we pop off the queue
        
        for neighbor in graph.neighbors(current_node):
            if neighbor == target:
                return path + [neighbor], nodes_explored
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
                
    return [], nodes_explored

@app.route('/')
def index():
    return render_template('index.html', nodes=nodes_list)

@app.route('/play')
def play():
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
    start_time_bfs = time.time()
    bfs_path, bfs_explored = custom_bfs(graph, start, target)
    bfs_time = time.time() - start_time_bfs

    # 2. Get AI Predicted Path
    
    node = start
    sl_path = []
    j = 0
    sl_explored = 0
    start_time_sl_ai = time.time()
    while j < 15:
        sl_path.append(node)
        sl_explored += 1 # The AI only explores the exact nodes it lands on
        if node == target:
            break
            
        neighbors = list(graph.neighbors(node))
        if not neighbors: 
            break

        try:
            links_str = ",".join(neighbors)
            links_encoded = link_encoder.transform([links_str])
            target_encoded = target_encoder.transform([target])

            predictions = model([links_encoded, target_encoded], training=False).numpy()[0]
            
            known_neighbors = [n for n in neighbors if n in choice_encoder.classes_]
            
            if known_neighbors:
                valid_ids = choice_encoder.transform(known_neighbors)
                best_valid_id = max(valid_ids, key=lambda idx: predictions[idx])
                node = choice_encoder.inverse_transform([best_valid_id])[0]
            else:
                predicted_label = np.argmax(predictions)
                node = choice_encoder.inverse_transform([predicted_label])[0]

        except Exception as e:
            print(f"Encoding Error (unseen node): {e}")
            break
            
        j += 1
        
    if sl_path[-1] != target:
        sl_path.append("(Failed)")

    sl_ai_time = time.time() - start_time_sl_ai

    # 3. Get Reinforcement Learning Predicted Path
    rl_path = []
    rl_explored = 0
    rl_time = 0.0
    if rl_model:
        node = start
        rl_path.append(node)
        j = 0
        start_time_rl_ai = time.time()
        while j < 15:
            rl_explored += 1 # The RL AI only explores the exact nodes it lands on
            if node == target:
                break
                
            neighbors = list(graph.neighbors(node))
            if not neighbors:
                break

            # 1. Create the observation array [Current ID, Target ID]
            obs = np.array([node_to_id[node], node_to_id[target]], dtype=np.int32)
            
            # 2. Create the mask (True for valid neighbors, False for everything else)
            mask = np.zeros(len(nodes_list), dtype=bool)
            for n in neighbors:
                mask[node_to_id[n]] = True
                
            # 3. Ask the RL AI for its absolute best move
            action, _states = rl_model.predict(obs, action_masks=mask, deterministic=True)
            
            # 4. Translate the integer ID back to a Wikipedia page name
            node = id_to_node[int(action)]
            rl_path.append(node)
            j += 1
            
        if rl_path[-1] != target:
            rl_path.append("(Failed)")
        rl_ai_time = time.time() - start_time_rl_ai
    else:
        rl_path = ["(RL Model Not Loaded)"]
    # Send the new stats to the frontend!
    return jsonify({
        #BFS stats
        "bfs_path": bfs_path,
        "bfs_time": round(bfs_time, 4),
        "bfs_explored": bfs_explored,
        #Supervised Learning AI stats
        "sl_path": sl_path,
        "sl_time": round(sl_ai_time, 4),
        "sl_explored": sl_explored,
        #Reinforcement Learning AI stats
        "rl_path": rl_path,
        "rl_time": round(rl_ai_time, 4),
        "rl_explored": rl_explored
    })

if __name__ == '__main__':
    app.run(debug=True)