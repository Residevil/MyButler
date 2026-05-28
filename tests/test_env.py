import sys
import os
import numpy as np
import torch

# Ensure butler package is visible to the test runner
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from butler.env import ButlerEnv
from butler.models import GenerativeModel

def test_environment_initialization():
    """Verify that ButlerEnv initializes correctly with correct state variables."""
    env = ButlerEnv()
    
    # Starting comfort variables should be 0.7
    np.testing.assert_array_almost_equal(env.essential, [0.7, 0.7, 0.7, 0.7])
    assert env.room_temp == 22.0
    assert env.current_music == 0
    assert env.time_of_day == 0.5
    
    # State representation length
    state = env.get_state()
    assert len(state) == 8

def test_environment_reset():
    """Verify reset function sets values randomly within proper ranges."""
    env = ButlerEnv()
    obs = env.reset()
    
    assert len(obs) == 8
    # Observations should be noisy but bounded
    assert 0.0 <= obs[0] <= 1.0
    assert 16.0 <= env.room_temp <= 30.0
    assert 0.3 <= env.time_of_day <= 0.75

def test_environment_step():
    """Verify that taking actions moves state variables logically."""
    env = ButlerEnv()
    env.reset()
    
    # Action 1: Heat up
    orig_target = env.target_temp
    obs, reward, done, info = env.step(1)
    
    assert env.target_temp == orig_target + 1.0
    assert len(obs) == 8
    assert not done

def test_generative_model_dimensions():
    """Verify that generative neural models compile and forward propagate correct dimensions."""
    model = GenerativeModel(state_dim=8, obs_dim=8, action_dim=9, essential_dim=4)
    
    # Mock inputs
    mock_state = torch.randn(1, 8)
    mock_action = torch.tensor([3], dtype=torch.long)
    
    next_state_mean, logvar, obs_pred, essential_pred = model.predict_next(mock_state, mock_action)
    
    assert next_state_mean.shape == (1, 8)
    assert logvar.shape == (1, 8)
    assert obs_pred.shape == (1, 8)
    assert essential_pred.shape == (1, 4)
    
    # Outputs of interoception should be bounded between 0 and 1 (sigmoid)
    assert torch.all(essential_pred >= 0.0)
    assert torch.all(essential_pred <= 1.0)
