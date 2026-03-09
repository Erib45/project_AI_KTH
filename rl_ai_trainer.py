import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
import random
import torch
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)

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
        
        self.valid_starts = [n for n in self.nodes if len(list(self.graph.neighbors(n))) > 0]

        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.MultiDiscrete([self.num_nodes, self.num_nodes])
        
        self.current_node = None
        self.target_node = None
        self.distance_map = {} # This will hold our BFS radar distances!
        self.steps_taken = 0
        self.max_steps = 20 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 🚨 NEW: Pick a target, then run Reverse BFS to map the whole maze instantly!
        self.target_node = random.choice(self.nodes)
        self.distance_map = nx.single_source_shortest_path_length(self.reverse_graph, self.target_node)
        
        # Find all nodes that actually have a valid path to this target
        winnable_starts = [n for n in self.distance_map.keys() if n != self.target_node and n in self.valid_starts]
        
        # If we picked a terrible target that nobody can reach, re-roll it
        while not winnable_starts:
            self.target_node = random.choice(self.nodes)
            self.distance_map = nx.single_source_shortest_path_length(self.reverse_graph, self.target_node)
            winnable_starts = [n for n in self.distance_map.keys() if n != self.target_node and n in self.valid_starts]

        # Spawn the AI on a guaranteed winnable page!
        self.current_node = random.choice(winnable_starts)
        self.steps_taken = 0
        
        return self._get_obs(), {}

    def step(self, action_id):
        self.steps_taken += 1
        chosen_node = self.id_to_node[action_id]
        
        neighbors = list(self.graph.neighbors(self.current_node))
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
            reward = 50
            terminated = True
            
        elif len(list(self.graph.neighbors(self.current_node))) == 0:
            reward = -30
            terminated = True
            
        elif self.steps_taken >= self.max_steps:
            reward = -2
            truncated = True
            
        else:
            # The AI is still playing. Let's grade its step using the radar!
            if new_distance < old_distance:
                reward = 5.0   # Amazing! It stepped toward the target.
            elif new_distance == old_distance:
                reward = 2    # Okay. It stepped sideways (neither closer nor further).
            else:
                reward = -2  # Terrible! It stepped backwards or off the path entirely.
                
            # Always subtract 1 point just to remind it that taking steps costs energy
            reward -= 1.0 

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([self.node_to_id[self.current_node], self.node_to_id[self.target_node]], dtype=np.int32)

    def valid_action_mask(self):
        mask = np.zeros(self.num_nodes, dtype=bool)
        neighbors = list(self.graph.neighbors(self.current_node))
        for n in neighbors: mask[self.node_to_id[n]] = True
        if len(neighbors) == 0: mask[self.node_to_id[self.current_node]] = True 
        return mask
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
    
    # Initialize the Environment
    env = WikiGraphEnv("wikipedia_subset_small.gml")
    
    # Wrap the environment to enforce the action masks
    env = ActionMasker(env, mask_fn)

    # Initialize the Maskable PPO Agent
    # We use an MLP (Multi-Layer Perceptron) policy for its brain
    print("Initializing the RL Agent...")
    model = MaskablePPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

    # Train the AI! (100,000 steps is a good quick test run)
    print("Beginning Training! The AI is now playing the game...")
    model.learn(total_timesteps=200_000)

    # Save the trained brain
    print("Training Complete! Saving the RL model...")
    model.save("rl_wiki_model")
    
    print("Saved as 'rl_wiki_model.zip'. Ready to load into the frontend!")

    # ... [Keep the model.save() line you already have] ...
    
    # ==========================================
    # 4. EVALUATE THE TRAINED AI (THE EXAM)
    # ==========================================
    print("\n==========================================")
    print("🤖 ADMINISTERING FINAL EXAM (100 GAMES)")
    print("==========================================")
    
    test_episodes = 100
    wins = 0
    total_steps_in_wins = 0
    
    for i in range(test_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done:
            # 1. Ask the environment which buttons are currently unlocked
            action_masks = env.unwrapped.valid_action_mask()
            
            # 2. Ask the AI for its prediction
            # deterministic=True tells the AI: "No random exploring! Pick your absolute best move."
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            # 3. Take the step
            obs, reward, terminated, truncated, info = env.step(action.item())
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