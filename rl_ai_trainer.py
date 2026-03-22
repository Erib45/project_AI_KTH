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
# 1. DEFINE THE GAME BOARD (ENVIRONMENT)
# ==========================================
class WikiGraphEnv(gym.Env):
    def __init__(self, graph_path="wikipedia_subset_small.gml"):
        super(WikiGraphEnv, self).__init__()
        
        print("Loading Graph for RL Environment...")
        self.graph = nx.read_gml(graph_path)
        
        # 🚨 NEW: Create a backwards graph so we can run Reverse BFS
        self.reverse_graph = self.graph.reverse()
        
        self.nodes = list(self.graph.nodes)
        self.num_nodes = len(self.nodes)
        
        self.node_to_id = {node: i for i, node in enumerate(self.nodes)}
        self.id_to_node = {i: node for i, node in enumerate(self.nodes)}
        
        # OPTIMIZATION 1: Precompute all neighbors first
        self.node_neighbors = {n: list(self.graph.neighbors(n)) for n in self.nodes}
        
        # OPTIMIZATION 2: Make this a SET instead of a LIST for instant O(1) lookups
        self.valid_starts = {n for n in self.nodes if len(self.node_neighbors[n]) > 0}
        
        # OPTIMIZATION 3: Precompute action masks for all 10,000 nodes so we don't build arrays during training
        self.node_masks = {}
        for node in self.nodes:
            mask = np.zeros(self.num_nodes, dtype=bool)
            neighbors = self.node_neighbors[node]
            for n in neighbors: mask[self.node_to_id[n]] = True
            if len(neighbors) == 0: mask[self.node_to_id[node]] = True
            self.node_masks[node] = mask

        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.MultiDiscrete([self.num_nodes, self.num_nodes])
        
        self.current_node = None
        self.target_node = None
        self.distance_map = {} # This will hold our BFS radar distances!
        self.steps_taken = 0
        self.max_steps = 20 
        
        # OPTIMIZATION: Cache BFS radar maps so we never calculate the same target twice!
        self.radar_cache = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 🚨 NEW: Pick a target, then run Reverse BFS to map the whole maze instantly!
        self.target_node = random.choice(self.nodes)
        
        if self.target_node not in self.radar_cache:
            self.radar_cache[self.target_node] = nx.single_source_shortest_path_length(self.reverse_graph, self.target_node)
        self.distance_map = self.radar_cache[self.target_node]
        
        # OPTIMIZATION 4: Blazing fast C-level set operations instead of python loops
        # (dict_keys naturally behave like sets in Python 3, so we can use '&' to find intersections!)
        winnable_starts = list(self.distance_map.keys() & self.valid_starts - {self.target_node})
        
        # If we picked a terrible target that nobody can reach, re-roll it
        while not winnable_starts:
            self.target_node = random.choice(self.nodes)
            if self.target_node not in self.radar_cache:
                self.radar_cache[self.target_node] = nx.single_source_shortest_path_length(self.reverse_graph, self.target_node)
            self.distance_map = self.radar_cache[self.target_node]
            winnable_starts = list(self.distance_map.keys() & self.valid_starts - {self.target_node})

        # Spawn the AI on a guaranteed winnable page!
        self.current_node = random.choice(winnable_starts)
        self.steps_taken = 0
        
        return self._get_obs(), {}

    def step(self, action_id):
        self.steps_taken += 1
        chosen_node = self.id_to_node[action_id]
        
        neighbors = self.node_neighbors[self.current_node]
        if chosen_node not in neighbors:
            return self._get_obs(), -50.0, True, False, {"msg": "Invalid Move!"}

        # 🚨 NEW: Check the BFS Radar before we move!
        old_distance = self.distance_map.get(self.current_node, float('inf'))
        new_distance = self.distance_map.get(chosen_node, float('inf'))
        
        # Move the AI
        self.current_node = chosen_node
        
        terminated = False
        truncated = False
        
        # 🚨 NEW: Your Custom Reward Scaling Logic!
        if self.current_node == self.target_node:
            reward = 100.0  # Jackpot! The AI found the target!
            terminated = True
            
        elif new_distance == float('inf'):
            reward = -10.0  # 🚨 THE ABYSS: The AI stepped onto a page with NO valid path to the target. Instant death!
            terminated = True
            
        elif len(self.node_neighbors[self.current_node]) == 0:
            reward = -10.0  # Oh no! The AI got stuck on a dead-end page with no links out.
            terminated = True
            
        elif self.steps_taken >= self.max_steps:
            reward = 0.0    # Time's up! (No penalty, just a neutral game over to avoid encouraging "suicide")
            truncated = True
            
        else:
            # The AI is still playing. Let's grade its step using the radar!
            if new_distance < old_distance:
                reward = 3.0   # Amazing! It stepped toward the target.
            elif new_distance == old_distance:
                reward = -1.0  # Okay. It stepped sideways (neither closer nor further).
            else:
                reward = -4.0  # Terrible! It stepped backwards or off the path entirely.

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([self.node_to_id[self.current_node], self.node_to_id[self.target_node]], dtype=np.int32)

    def valid_action_mask(self):
        # OPTIMIZATION 5: Instant O(1) dictionary lookup
        return self.node_masks[self.current_node]

# ==========================================
# 1.5 CUSTOM EMBEDDING EXTRACTOR
# ==========================================
class GraphNodeExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.MultiDiscrete, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        num_nodes = observation_space.nvec[0]
        self.embedding = nn.Embedding(num_nodes, features_dim // 2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs_ints = observations.long()
        current_embed = self.embedding(obs_ints[:, 0])
        target_embed = self.embedding(obs_ints[:, 1])
        return torch.cat([current_embed, target_embed], dim=1)
        
# ==========================================
# 2. HELPER FUNCTION TO APPLY THE MASK
# ==========================================
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()


# ==========================================
# 3. RUN THE TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    print("Setting up the Wikipedia Arena...")
    
    num_parallel_games = 6  # Spawns 6 parallel dimensions of the game!
    
    # We create a factory function that generates identical clones of our environment
    def make_env():
        def _init():
            # Each worker gets its own isolated graph and action masker
            e = WikiGraphEnv("wikipedia_subset_small.gml")
            e = ActionMasker(e, mask_fn)
            return e
        return _init

    # Initialize the Vectorized Environment (SubprocVecEnv uses true CPU multiprocessing)
    print(f"Spinning up {num_parallel_games} CPU worker cores...")
    vec_env = SubprocVecEnv([make_env() for _ in range(num_parallel_games)])
    vec_env = VecMonitor(vec_env) # <--- PROPER WAY TO MONITOR VECTOR ENVS!

    # Initialize the Maskable PPO Agent
    # We use an MLP (Multi-Layer Perceptron) policy for its brain
    print("Initializing the RL Agent...")
    policy_kwargs = dict(
        features_extractor_class=GraphNodeExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    model = MaskablePPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.0003)

    # Train the AI!
    print("Beginning Training! The AI is now playing the game...")
    model.learn(total_timesteps=10_000_000)

    # Save the trained brain
    print("Training Complete! Saving the RL model...")
    model.save("rl_wiki_model")
    
    print("Saved as 'rl_wiki_model.zip'. Ready to load into the frontend!")
    
    # ==========================================
    # CLOSE THE VECTORIZED ENVIRONMENTS
    vec_env.close()
    
    # ==========================================
    # 4. EVALUATE THE TRAINED AI (THE EXAM)
    # ==========================================
    print("\n==========================================")
    print("🤖 ADMINISTERING FINAL EXAM (100 GAMES)")
    print("==========================================")
    
    test_episodes = 100
    wins = 0
    total_steps_in_wins = 0
    
    # Create a single, standard environment for the exam
    eval_env = ActionMasker(WikiGraphEnv("wikipedia_subset_small.gml"), mask_fn)
    
    for i in range(test_episodes):
        obs, _ = eval_env.reset()
        done = False
        steps = 0
        
        while not done:
            # 1. Ask the environment which buttons are currently unlocked
            action_masks = eval_env.unwrapped.valid_action_mask()
            
            # 2. Ask the AI for its prediction
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            # 3. Take the step
            obs, reward, terminated, truncated, info = eval_env.step(action.item())
            steps += 1
            
            # 4. Check if the game ended
            if terminated or truncated:
                done = True
                # In our reward system, exactly 100.0 means it hit the target!
                if reward == 100.0: 
                    wins += 1
                    total_steps_in_wins += steps

    # Calculate final grades
    win_rate = (wins / test_episodes) * 100
    avg_steps = (total_steps_in_wins / wins) if wins > 0 else 0
    
    print(f"🎯 Games Played:  {test_episodes}")
    print(f"🏆 Win Rate:      {win_rate:.1f}%")
    print(f"👟 Avg Steps:     {avg_steps:.1f} (when successful)")
    print("==========================================\n")