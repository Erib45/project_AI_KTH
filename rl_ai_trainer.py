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
        
        # Load the graph
        print("Loading Graph for RL Environment...")
        self.graph = nx.read_gml(graph_path)
        self.nodes = list(self.graph.nodes)
        self.num_nodes = len(self.nodes)
        
        # Create dictionaries to translate Node Names (strings) <--> Node IDs (integers)
        # RL algorithms only understand numbers!
        self.node_to_id = {node: i for i, node in enumerate(self.nodes)}
        self.id_to_node = {i: node for i, node in enumerate(self.nodes)}
        
        # Pre-calculate valid start nodes (must have at least 1 outgoing link)
        self.valid_starts = [n for n in self.nodes if len(list(self.graph.neighbors(n))) > 0]

        # --- THE SPACES ---
        # Action Space: The AI has a massive controller with a button for EVERY page.
        self.action_space = spaces.Discrete(self.num_nodes)
        
        # Observation Space: The AI needs to know [Current Node ID, Target Node ID]
        self.observation_space = spaces.MultiDiscrete([self.num_nodes, self.num_nodes])
        
        # Game State Variables
        self.current_node = None
        self.target_node = None
        self.steps_taken = 0
        self.max_steps = 20 # Force the game to end if the AI wanders too far

    def reset(self, seed=None, options=None):
        """Starts a new game round."""
        super().reset(seed=seed)
        
        # Pick a safe random start and target
        self.current_node = random.choice(self.valid_starts)
        self.target_node = random.choice(self.nodes)
        while self.current_node == self.target_node:
            self.target_node = random.choice(self.nodes)
            
        self.steps_taken = 0
        
        # Return the observation [Current ID, Target ID] and an empty info dict
        obs = np.array([self.node_to_id[self.current_node], self.node_to_id[self.target_node]], dtype=np.int32)
        return obs, {}

    def step(self, action_id):
        """The AI presses a button (takes an action)."""
        self.steps_taken += 1
        chosen_node = self.id_to_node[action_id]
        
        # 1. Check if the AI tried to cheat (should be prevented by the mask, but just in case!)
        neighbors = list(self.graph.neighbors(self.current_node))
        if chosen_node not in neighbors:
            # Massive penalty for breaking the rules, and instantly end the game
            return self._get_obs(), -50.0, True, False, {"msg": "Invalid Move!"}

        # 2. Move the AI to the new room
        self.current_node = chosen_node
        
        # 3. Calculate Rewards!
        terminated = False
        truncated = False
        reward = -1.0 # -1 point for every step (encourages speed)
        
        if self.current_node == self.target_node:
            reward = 200.0 # +200 points for winning!
            terminated = True
            
        elif len(list(self.graph.neighbors(self.current_node))) == 0:
            reward = -50.0 # -30 points for walking into a dead end
            terminated = True
            
        elif self.steps_taken >= self.max_steps:
            reward = -5.0 # Small penalty for timing out
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        """Helper to format the observation array."""
        return np.array([self.node_to_id[self.current_node], self.node_to_id[self.target_node]], dtype=np.int32)

    def valid_action_mask(self):
        """
        THE SECRET WEAPON: This creates an array of True/False for all nodes.
        True = The button is unlocked (it is a valid neighbor).
        False = The button is physically locked by the wrapper.
        """
        mask = np.zeros(self.num_nodes, dtype=bool)
        neighbors = list(self.graph.neighbors(self.current_node))
        for n in neighbors:
            mask[self.node_to_id[n]] = True
            
        # If dead end, just unlock the current node so it doesn't crash before the game ends
        if len(neighbors) == 0:
            mask[self.node_to_id[self.current_node]] = True 
            
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
            obs, reward, terminated, truncated, info = env.step(action)
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