import numpy as np
import torch
import torch.nn as nn
import random

class ActiveInferenceAgent:
    """
    Active Inference Agent representing the predictive, homeostatic mind of MyButler.
    Continuous online learning, online sensory belief correction, and
    Expected Free Energy minimization (balancing survival and curiosity drives).
    """
    def __init__(self, state_dim=8, obs_dim=8, action_dim=9, essential_dim=4):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.essential_dim = essential_dim
        
        # 1. Initialize Generative Model
        from butler.models import GenerativeModel
        self.gen_model = GenerativeModel(state_dim, obs_dim, action_dim, essential_dim)
        
        # 2. Current belief about hidden state (s_t)
        self.belief_state = torch.zeros(state_dim, dtype=torch.float32)
        
        # 3. Preference prior over essential variables (target high-evidence well-being)
        self.preferred_essential = torch.tensor([0.85, 0.85, 0.85, 0.85], dtype=torch.float32)
        
        # 4. Replay Buffer for online training
        self.buffer = []
        self.buffer_max_size = 2000
        self.batch_size = 32
        
        # 5. Exploration/Curiosity parameters
        self.beta = 0.15                # Weight of curiosity drive (epistemic value)
        
    def reset(self, initial_obs):
        """Resets the agent's belief state to match initial observations."""
        self.belief_state = torch.tensor(initial_obs.copy(), dtype=torch.float32)
        return self.belief_state.numpy()
        
    def store_transition(self, s, a, ns, o, e):
        """Stores experience tuple in the replay buffer for continuous training."""
        self.buffer.append((s, a, ns, o, e))
        if len(self.buffer) > self.buffer_max_size:
            self.buffer.pop(0)

    def update_belief(self, observation, last_action=None):
        """
        Sensory Correction / Online Belief Inference.
        Updates the internal state estimation (belief_state) to minimize 
        Variational Free Energy (difference between expected and actual sensory inputs).
        """
        self.gen_model.eval()
        
        # We perform a few steps of gradient descent on belief state
        s_belief = self.belief_state.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([s_belief], lr=0.1)
        
        obs_t = torch.tensor(observation, dtype=torch.float32)
        
        # Number of belief adjustment iterations (predictive coding correction steps)
        for _ in range(5):
            optimizer.zero_grad()
            
            # Predict observation from current belief
            obs_pred = self.gen_model.observation_model(s_belief)
            
            # Variational Free Energy bounds:
            # 1. Sensory prediction error (accuracy term)
            sensory_error = torch.mean((obs_pred - obs_t)**2)
            
            # 2. Dynamic transition prediction error (complexity term, if previous step exists)
            transition_error = 0.0
            if last_action is not None:
                # Expect state to match previous transition prediction
                prev_state_pred, logvar = self.gen_model.transition_model(self.belief_state, torch.tensor([last_action]))
                transition_error = torch.mean(torch.exp(-logvar) * (s_belief - prev_state_pred)**2)
                
            loss = sensory_error + 0.1 * transition_error
            loss.backward()
            optimizer.step()
            
        self.belief_state = s_belief.clone().detach()
        # Ensure belief variables that are homeostatic stay bounded
        self.belief_state[:4] = torch.clip(self.belief_state[:4], 0.0, 1.0)
        
        return self.belief_state.numpy()

    def select_action(self):
        """
        Model-Based Planning via Expected Free Energy (EFE) minimization.
        For each candidate action, evaluate predicted homeostatic outcomes (survival)
        and uncertainty reduction (curiosity).
        """
        self.gen_model.eval()
        
        best_action = 0
        lowest_efe = float('inf')
        
        efe_breakdown = {}
        
        # Evaluate all possible actions
        for action in range(self.action_dim):
            with torch.no_grad():
                # 1. Predict next state s_{t+1} and its log-variance
                next_state_mean, logvar = self.gen_model.transition_model(self.belief_state, torch.tensor([action]))
                
                # 2. Predict next essential variables e_{t+1}
                essential_pred = self.gen_model.interoceptive_model(next_state_mean).squeeze(0)
                
                # 3. Compute Survival Cost (Pragmatic Value)
                # Deviation from optimal comfort level
                survival_cost = torch.mean((essential_pred - self.preferred_essential)**2).item()
                
                # 4. Compute Epistemic Curiosity Value
                # Model uncertainty is represented by the transition log-variance.
                # Epistemic value is the predicted standard deviation (average uncertainty).
                uncertainty = torch.mean(torch.exp(0.5 * logvar)).item()
                epistemic_value = uncertainty
                
                # 5. Expected Free Energy = Survival_Cost - beta * Epistemic_Value
                # If homeostatic variables are dangerously low (high deviation), survival cost dominates.
                # If safe (> 0.75), curiosity drive drives active learning.
                current_health_avg = torch.mean(self.belief_state[:4]).item()
                
                # Dynamic modulation of curiosity based on homeostatic security
                current_beta = self.beta if current_health_avg > 0.7 else 0.01
                
                efe = survival_cost - current_beta * epistemic_value
                efe_breakdown[action] = (survival_cost, epistemic_value, efe)
                
                if efe < lowest_efe:
                    lowest_efe = efe
                    best_action = action
                    
        # Small exploration noise (epsilon-greedy equivalent, but biologically styled as action selection noise)
        if random.random() < 0.05:
            best_action = random.randint(0, self.action_dim - 1)
            
        return best_action, efe_breakdown[best_action]

    def learn(self):
        """
        Continuously train the generative models online 
        using experiences sampled from the replay buffer.
        """
        if len(self.buffer) < self.batch_size:
            return None
            
        # Sample random batch
        batch = random.sample(self.buffer, self.batch_size)
        
        states = np.array([x[0] for x in batch], dtype=np.float32)
        actions = np.array([x[1] for x in batch], dtype=np.int64)
        next_states = np.array([x[2] for x in batch], dtype=np.float32)
        observations = np.array([x[3] for x in batch], dtype=np.float32)
        essentials = np.array([x[4] for x in batch], dtype=np.float32)
        
        # Execute gradient update step
        losses = self.gen_model.train_step(states, actions, next_states, observations, essentials)
        return losses
