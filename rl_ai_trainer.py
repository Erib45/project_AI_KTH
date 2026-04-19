import os
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
import random
import torch
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# ==========================================
# 0. PRE-TRAIN NODE EMBEDDINGS (Node2Vec)
# ==========================================
def precompute_node2vec_embeddings(graph_path, embed_dim=64, cache_path="node2vec_embeddings.npy"):
    if os.path.exists(cache_path):
        print("Loading cached Node2Vec embeddings...")
        return np.load(cache_path)

    print("Computing Node2Vec embeddings (this may take a few minutes)...")
    from node2vec import Node2Vec
    graph = nx.read_gml(graph_path)
    nodes = list(graph.nodes)
    node_to_id = {node: i for i, node in enumerate(nodes)}

    n2v_model = Node2Vec(graph, dimensions=embed_dim, walk_length=30, num_walks=100, workers=4, quiet=True)
    n2v = n2v_model.fit(window=10, min_count=1, batch_words=4)

    embeddings = np.zeros((len(nodes), embed_dim), dtype=np.float32)
    for node in nodes:
        key = str(node)
        if key in n2v.wv:
            embeddings[node_to_id[node]] = n2v.wv[key]

    np.save(cache_path, embeddings)
    print(f"Node2Vec embeddings cached to '{cache_path}'")
    return embeddings

# ==========================================
# 1. DEFINE THE GAME BOARD (ENVIRONMENT)
# ==========================================
class WikiGraphEnv(gym.Env):
    def __init__(self, graph_path="wikipedia_subset_small.gml"):
        super(WikiGraphEnv, self).__init__()

        print("Loading Graph for RL Environment...")
        self.graph = nx.read_gml(graph_path)

        self.reverse_graph = self.graph.reverse()

        self.nodes = list(self.graph.nodes)
        self.num_nodes = len(self.nodes)

        self.node_to_id = {node: i for i, node in enumerate(self.nodes)}
        self.id_to_node = {i: node for i, node in enumerate(self.nodes)}

        # OPTIMIZATION 1: Precompute all neighbors first
        self.node_neighbors = {n: list(self.graph.neighbors(n)) for n in self.nodes}

        # OPTIMIZATION 2: Make this a SET instead of a LIST for instant O(1) lookups
        self.valid_starts = {n for n in self.nodes if len(self.node_neighbors[n]) > 0}

        # OPTIMIZATION 3: Precompute action masks for all nodes
        self.node_masks = {}
        for node in self.nodes:
            mask = np.zeros(self.num_nodes, dtype=bool)
            neighbors = self.node_neighbors[node]
            for n in neighbors: mask[self.node_to_id[n]] = True
            if len(neighbors) == 0: mask[self.node_to_id[node]] = True
            self.node_masks[node] = mask

        # max_steps must be set before observation_space so we can use it in the space definition
        self.max_steps = 40
        self.action_space = spaces.Discrete(self.num_nodes)
        # Observation: [current_node_id, target_node_id, bfs_distance_to_target]
        # Distance ranges 0..max_steps+1, where max_steps+1 means "unreachable"
        self.observation_space = spaces.MultiDiscrete([self.num_nodes, self.num_nodes, self.max_steps + 2])

        self.current_node = None
        self.target_node = None
        self.distance_map = {}
        self.steps_taken = 0

        # OPTIMIZATION: Cache BFS radar maps so we never calculate the same target twice!
        self.radar_cache = {}
        self.visited_nodes = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.target_node = random.choice(self.nodes)

        if self.target_node not in self.radar_cache:
            self.radar_cache[self.target_node] = nx.single_source_shortest_path_length(self.reverse_graph, self.target_node)
        self.distance_map = self.radar_cache[self.target_node]

        # OPTIMIZATION 4: Blazing fast C-level set operations
        winnable_starts = list(self.distance_map.keys() & self.valid_starts - {self.target_node})

        while not winnable_starts:
            self.target_node = random.choice(self.nodes)
            if self.target_node not in self.radar_cache:
                self.radar_cache[self.target_node] = nx.single_source_shortest_path_length(self.reverse_graph, self.target_node)
            self.distance_map = self.radar_cache[self.target_node]
            winnable_starts = list(self.distance_map.keys() & self.valid_starts - {self.target_node})

        self.current_node = random.choice(winnable_starts)
        self.steps_taken = 0
        self.visited_nodes = {self.current_node}

        return self._get_obs(), {}

    def step(self, action_id):
        self.steps_taken += 1
        chosen_node = self.id_to_node[action_id]

        neighbors = self.node_neighbors[self.current_node]
        if chosen_node not in neighbors:
            return self._get_obs(), -50.0, True, False, {"msg": "Invalid Move!"}

        old_distance = self.distance_map.get(self.current_node, float('inf'))
        new_distance = self.distance_map.get(chosen_node, float('inf'))

        self.current_node = chosen_node

        terminated = False
        truncated = False

        if self.current_node == self.target_node:
            reward = 100.0
            terminated = True

        elif self.current_node in self.visited_nodes:
            reward = -10.0  # walked in a circle

        elif new_distance == float('inf'):
            reward = -10.0  # stepped onto a page with no path to target
            terminated = True

        elif len(self.node_neighbors[self.current_node]) == 0:
            reward = -10.0  # dead-end page with no links out
            terminated = True

        elif self.steps_taken >= self.max_steps:
            reward = 0.0
            truncated = True

        else:
            self.visited_nodes.add(self.current_node)
            if new_distance < old_distance:
                reward = 3.0
            elif new_distance == old_distance:
                reward = -1.0
            else:
                reward = -4.0

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        dist = self.distance_map.get(self.current_node, self.max_steps + 1)
        dist = min(int(dist), self.max_steps + 1)
        return np.array([
            self.node_to_id[self.current_node],
            self.node_to_id[self.target_node],
            dist
        ], dtype=np.int32)

    def valid_action_mask(self):
        # OPTIMIZATION 5: Instant O(1) dictionary lookup
        return self.node_masks[self.current_node]

# ==========================================
# 1.5 CUSTOM EMBEDDING EXTRACTOR
# ==========================================
class GraphNodeExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.MultiDiscrete, features_dim: int = 129, pretrained_embeddings=None):
        super().__init__(observation_space, features_dim)
        num_nodes = observation_space.nvec[0]
        # embed_dim * 2 + 1 (distance scalar) == features_dim
        embed_dim = (features_dim - 1) // 2  # 64 when features_dim=129
        self.max_dist = float(observation_space.nvec[2] - 1)
        self.embedding = nn.Embedding(num_nodes, embed_dim)

        if pretrained_embeddings is not None:
            weight = torch.FloatTensor(pretrained_embeddings)
            # Pad or truncate columns to match embed_dim
            if weight.shape[1] < embed_dim:
                padding = torch.zeros(weight.shape[0], embed_dim - weight.shape[1])
                weight = torch.cat([weight, padding], dim=1)
            elif weight.shape[1] > embed_dim:
                weight = weight[:, :embed_dim]
            self.embedding.weight.data.copy_(weight)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs_ints = observations.long()
        current_embed = self.embedding(obs_ints[:, 0])
        target_embed = self.embedding(obs_ints[:, 1])
        # Normalize distance to [0, 1] so it's on the same scale as embeddings
        distance_norm = observations[:, 2:3].float() / self.max_dist
        return torch.cat([current_embed, target_embed, distance_norm], dim=1)

# ==========================================
# 2. HELPER FUNCTION TO APPLY THE MASK
# ==========================================
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()


# ==========================================
# 3. RUN THE TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    GRAPH_PATH = "wikipedia_subset_small.gml"
    EMBED_DIM = 64

    print("Setting up the Wikipedia Arena...")

    # Pre-train node embeddings with Node2Vec before any environment is created.
    # Results are cached to disk so subsequent runs skip recomputation.
    # Install dependency: pip install node2vec
    pretrained_embeddings = precompute_node2vec_embeddings(GRAPH_PATH, embed_dim=EMBED_DIM)

    num_parallel_games = 6

    def make_env():
        def _init():
            e = WikiGraphEnv(GRAPH_PATH)
            e = ActionMasker(e, mask_fn)
            return e
        return _init

    print(f"Spinning up {num_parallel_games} CPU worker cores...")
    vec_env = SubprocVecEnv([make_env() for _ in range(num_parallel_games)])
    vec_env = VecMonitor(vec_env)

    print("Initializing the RL Agent...")
    policy_kwargs = dict(
        features_extractor_class=GraphNodeExtractor,
        # features_dim=129: 64 (current embed) + 64 (target embed) + 1 (distance scalar)
        features_extractor_kwargs=dict(features_dim=129, pretrained_embeddings=pretrained_embeddings),
    )
    model = MaskablePPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.0003)

    print("Beginning Training! The AI is now playing the game...")
    model.learn(total_timesteps=100_000)

    print("Training Complete! Saving the RL model...")
    model.save("rl_wiki_model")

    print("Saved as 'rl_wiki_model.zip'. Ready to load into the frontend!")

    vec_env.close()

    # ==========================================
    # 4. EVALUATE THE TRAINED AI (THE EXAM)
    # ==========================================
    print("\n==========================================")
    print("ADMINISTERING FINAL EXAM (100 GAMES)")
    print("==========================================")

    test_episodes = 100
    wins = 0
    total_steps_in_wins = 0

    eval_env = ActionMasker(WikiGraphEnv(GRAPH_PATH), mask_fn)

    for i in range(test_episodes):
        obs, _ = eval_env.reset()
        done = False
        steps = 0

        while not done:
            action_masks = eval_env.unwrapped.valid_action_mask()
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action.item())
            steps += 1

            if terminated or truncated:
                done = True
                if terminated and reward == 100.0:
                    wins += 1
                    total_steps_in_wins += steps

    win_rate = (wins / test_episodes) * 100
    avg_steps = (total_steps_in_wins / wins) if wins > 0 else 0

    print(f"Games Played:  {test_episodes}")
    print(f"Win Rate:      {win_rate:.1f}%")
    print(f"Avg Steps:     {avg_steps:.1f} (when successful)")
    print("==========================================\n")
