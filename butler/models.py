import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbabilisticTransitionModel(nn.Module):
    """
    Predicts the mean and log-variance of the next state s_{t+1} 
    given current state s_t and action a_t.
    Uses a residual connection: s_{t+1} = s_t + Delta_s.
    """
    def __init__(self, state_dim=8, action_dim=9, hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Action embedding layer
        self.action_embed = nn.Embedding(action_dim, 16)
        
        # Core layers
        self.fc1 = nn.Linear(state_dim + 16, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Head for transition mean (Delta_s)
        self.mean_head = nn.Linear(hidden_dim, state_dim)
        # Head for transition log-variance (model uncertainty)
        self.logvar_head = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, state, action):
        # Ensure state and action are tensors with batch dimension
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long)
            
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 0 or len(action.shape) == 1 and action.shape[0] == 1:
            action = action.view(-1)
            
        # Get action embedding and concatenate with state
        act_emb = self.action_embed(action)
        x = torch.cat([state, act_emb], dim=-1)
        
        # Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean_delta = self.mean_head(x)
        logvar = torch.clamp(self.logvar_head(x), min=-10.0, max=2.0)  # Bound variance to avoid infinity
        
        # Residual transition: s_{t+1} = s_t + Delta_s
        next_state_mean = state + mean_delta
        
        return next_state_mean, logvar

class ObservationModel(nn.Module):
    """
    Predicts the expected sensory observation o_t given the hidden state s_t.
    """
    def __init__(self, state_dim=8, obs_dim=8, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, obs_dim)
        
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        x = F.relu(self.fc1(state))
        obs_pred = self.fc2(x)
        return obs_pred

class InteroceptiveModel(nn.Module):
    """
    Predicts the human's essential well-being variables e_t 
    (Comfort, Health, Calm, Engagement) from the hidden state s_t.
    These are bounded between 0.0 and 1.0.
    """
    def __init__(self, state_dim=8, essential_dim=4, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, essential_dim)
        
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        x = F.relu(self.fc1(state))
        # Use sigmoid to bound outputs between 0 and 1
        essential_pred = torch.sigmoid(self.fc2(x))
        return essential_pred

class GenerativeModel(nn.Module):
    """
    Unified Generative Model.
    Wraps the transition, observation, and interoceptive models.
    Provides utility methods to compute prediction errors (free energy bounds)
    and perform training updates.
    """
    def __init__(self, state_dim=8, obs_dim=8, action_dim=9, essential_dim=4):
        super().__init__()
        self.transition_model = ProbabilisticTransitionModel(state_dim, action_dim)
        self.observation_model = ObservationModel(state_dim, obs_dim)
        self.interoceptive_model = InteroceptiveModel(state_dim, essential_dim)
        
        # Optimizer Setup
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
        
    def predict_next(self, state, action):
        """
        Predicts next state (mean and logvar), next observation, and next essential variables.
        """
        self.eval()
        with torch.no_grad():
            next_state_mean, logvar = self.transition_model(state, action)
            obs_pred = self.observation_model(next_state_mean)
            essential_pred = self.interoceptive_model(next_state_mean)
            
        return next_state_mean, logvar, obs_pred, essential_pred

    def compute_loss(self, state, action, next_state, obs, essential):
        """
        Computes prediction loss over a batch.
        Transition loss uses negative log-likelihood under predicted Gaussian distribution.
        """
        # 1. Forward predictions
        next_state_mean, logvar = self.transition_model(state, action)
        obs_pred = self.observation_model(next_state)
        essential_pred = self.interoceptive_model(next_state)
        
        # 2. Gaussian NLL Loss for transition:
        # Loss = 0.5 * exp(-logvar) * (next_state - mean)^2 + 0.5 * logvar
        inv_var = torch.exp(-logvar)
        transition_loss = torch.mean(0.5 * inv_var * (next_state - next_state_mean)**2 + 0.5 * logvar)
        
        # 3. MSE Loss for observation and interoception predictions
        obs_loss = F.mse_loss(obs_pred, obs)
        essential_loss = F.mse_loss(essential_pred, essential)
        
        total_loss = transition_loss + obs_loss + 2.0 * essential_loss
        return total_loss, {
            "total": total_loss.item(),
            "transition": transition_loss.item(),
            "obs": obs_loss.item(),
            "essential": essential_loss.item()
        }

    def train_step(self, states, actions, next_states, observations, essentials):
        """Performs a single gradient descent update step."""
        self.train()
        self.optimizer.zero_grad()
        
        # Convert inputs to torch tensors
        s = torch.tensor(states, dtype=torch.float32)
        a = torch.tensor(actions, dtype=torch.long)
        ns = torch.tensor(next_states, dtype=torch.float32)
        o = torch.tensor(observations, dtype=torch.float32)
        e = torch.tensor(essentials, dtype=torch.float32)
        
        loss, losses_dict = self.compute_loss(s, a, ns, o, e)
        loss.backward()
        self.optimizer.step()
        
        return losses_dict
