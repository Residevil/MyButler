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

        # Essential variable indices in state / observation vectors
        self.ESSENTIAL_NAMES = ("comfort", "health", "calm", "engagement")
        self.CALM_IDX = 2
        
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

        Returns (belief_numpy, diagnostics_dict) for logging inference quality.
        """
        self.gen_model.eval()
        belief_pre = self.belief_state.clone()
        calm_idx = self.CALM_IDX
        obs_t = torch.tensor(observation, dtype=torch.float32)
        sensory_error_calm_final = 0.0

        with torch.no_grad():
            pred_calm_pre = self.gen_model.observation_model(belief_pre).squeeze()[calm_idx].item()
            intero_calm_pre = self.gen_model.interoceptive_model(belief_pre).squeeze()[calm_idx].item()

            transition_pred_calm = None
            transition_logvar_calm = None
            if last_action is not None:
                next_mean, logvar = self.gen_model.transition_model(
                    belief_pre, torch.tensor([last_action])
                )
                transition_pred_calm = next_mean[0, calm_idx].item()
                transition_logvar_calm = logvar[0, calm_idx].item()

        s_belief = belief_pre.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([s_belief], lr=0.1)

        for _ in range(5):
            optimizer.zero_grad()
            obs_pred = self.gen_model.observation_model(s_belief).squeeze(0)
            sensory_error = torch.mean((obs_pred - obs_t) ** 2)
            sensory_error_calm_final = ((obs_pred[calm_idx] - obs_t[calm_idx]) ** 2).item()

            transition_error = 0.0
            if last_action is not None and transition_pred_calm is not None:
                prev_state_pred, logvar = self.gen_model.transition_model(
                    belief_pre, torch.tensor([last_action])
                )
                transition_error = torch.mean(
                    torch.exp(-logvar) * (s_belief - prev_state_pred) ** 2
                )

            loss = sensory_error + 0.1 * transition_error
            loss.backward()
            optimizer.step()

        belief_calm_raw = s_belief[calm_idx].item()
        self.belief_state = s_belief.clone().detach()
        self.belief_state[:4] = torch.clip(self.belief_state[:4], 0.0, 1.0)
        belief_calm = self.belief_state[calm_idx].item()

        with torch.no_grad():
            obs_pred = self.gen_model.observation_model(self.belief_state).squeeze(0)
            intero_pred = self.gen_model.interoceptive_model(self.belief_state).squeeze(0)

        diagnostics = {
            "belief_calm": belief_calm,
            "belief_calm_pre": belief_pre[calm_idx].item(),
            "belief_calm_raw": belief_calm_raw,
            "belief_calm_delta": belief_calm - belief_pre[calm_idx].item(),
            "belief_calm_pinned": (
                abs(belief_calm_raw - belief_calm) > 1e-4
                or belief_calm <= 1e-4
                or belief_calm >= 1.0 - 1e-4
            ),
            "predicted_calm_obs_pre": pred_calm_pre,
            "predicted_calm_obs": obs_pred[calm_idx].item(),
            "predicted_calm_intero_pre": intero_calm_pre,
            "predicted_calm_intero": intero_pred[calm_idx].item(),
            "obs_calm": observation[calm_idx],
            "obs_error_calm_vs_obs": (obs_pred[calm_idx] - obs_t[calm_idx]).item(),
            "obs_error_calm_vs_true": None,
            "intero_error_calm_vs_true": None,
            "sensory_error_calm": sensory_error_calm_final,
            "transition_pred_calm": transition_pred_calm,
            "transition_logvar_calm": transition_logvar_calm,
        }
        return self.belief_state.numpy(), diagnostics

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
