import numpy as np

class ButlerEnv:
    """
    MyButler Homeostatic Environment.
    Simulates a human's biological and emotional well-being variables under
    environmental factors and butler interventions (temperature, music, speech).
    
    The state is hidden/partially observed, and actions drive homeostatic transitions.
    """
    def __init__(self):
        # 1. Essential variables: Comfort, Health, Calm, Engagement (0.0 to 1.0)
        # Optimal levels are around 0.85+
        self.essential = np.array([0.7, 0.7, 0.7, 0.7], dtype=np.float32)
        
        # 2. Environmental states
        self.room_temp = 22.0          # in Celsius
        self.target_temp = 22.0        # target thermostat setting
        self.time_of_day = 0.5         # 0.0 (midnight) to 1.0 (next midnight)
        self.current_music = 0         # 0: None, 1: Classical, 2: Jazz, 3: Pop, 4: Ambient
        self.last_action = 0           # Keep track of last action taken
        self.steps_since_speech = 0
        
        # 3. Dynamic parameters
        self.dt = 0.05                 # Fraction of an hour per step (3 minutes)
        
    def get_state(self):
        """Returns the true internal state of the human and environment."""
        return np.concatenate([
            self.essential,
            [self.room_temp, self.target_temp, self.time_of_day, float(self.current_music)]
        ])

    def reset(self):
        """Resets the environment to initial noisy homeostatic conditions."""
        self.essential = np.array([0.7, 0.7, 0.7, 0.7], dtype=np.float32)
        self.room_temp = 20.0 + np.random.rand() * 4.0   # 20 to 24 C
        self.target_temp = self.room_temp
        self.time_of_day = 0.3 + np.random.rand() * 0.4  # morning/afternoon
        self.current_music = 0
        self.last_action = 0
        self.steps_since_speech = 0
        return self._get_obs()

    def _get_obs(self):
        """
        Sensory perception: Noisy, sparse observation of the human variables 
        along with precise physical measurements.
        """
        noise = np.random.normal(0.0, 0.03, size=4).astype(np.float32)
        noisy_essential = np.clip(self.essential + noise, 0.0, 1.0)
        
        # Observation vector: [noisy_comfort, noisy_health, noisy_calm, noisy_engagement,
        #                      room_temp, target_temp, time_of_day, music]
        return np.concatenate([
            noisy_essential,
            [self.room_temp, self.target_temp, self.time_of_day, float(self.current_music)]
        ]).astype(np.float32)

    def step(self, action):
        """
        Transitions the environment one step forward given an action.
        
        Actions:
        0: Do nothing
        1: Increase target temperature (+1.0 C)
        2: Decrease target temperature (-1.0 C)
        3: Play Classical music (soothing)
        4: Play Jazz music (stimulating/smooth)
        5: Play Pop music (energetic)
        6: Play Ambient music (background comforting)
        7: Speak comforting phrase (increases calm/comfort)
        8: Speak stimulating/encouraging phrase (increases engagement)
        """
        self.last_action = action
        self.steps_since_speech += 1
        
        # 1. Apply Action Effects
        if action == 1:
            self.target_temp = min(30.0, self.target_temp + 1.0)
        elif action == 2:
            self.target_temp = max(16.0, self.target_temp - 1.0)
        elif 3 <= action <= 6:
            self.current_music = action - 2  # 1: Classical, 2: Jazz, etc.
        elif action == 7:
            self.steps_since_speech = 0
        elif action == 8:
            self.steps_since_speech = 0
            
        # 2. Physics: Room temperature moves toward target temperature
        temp_drift = (self.target_temp - self.room_temp) * 0.15
        self.room_temp += temp_drift + np.random.normal(0, 0.05)
        
        # 3. Simulate Human Natural Drift & Action Impacts
        
        # -- COMFORT --
        # Comfort drops if temperature deviates from human ideal (22.0 C)
        temp_discomfort = np.abs(self.room_temp - 22.0)
        comfort_drift = -0.05 * temp_discomfort
        
        # Ambient music (genre 4) or comfort phrase (action 7) boosts comfort slightly
        comfort_boost = 0.0
        if self.current_music == 4:
            comfort_boost += 0.02
        if action == 7:
            comfort_boost += 0.08
            
        self.essential[0] += self.dt * (comfort_drift + comfort_boost)
        
        # -- HEALTH --
        # Health drifts down slowly naturally, restored by comfort and nighttime rest
        is_night = (self.time_of_day < 0.25) or (self.time_of_day > 0.75)
        sleep_boost = 0.06 if is_night else -0.01
        
        health_drift = -0.02 + 0.04 * self.essential[0] + sleep_boost
        self.essential[1] += self.dt * health_drift
        
        # -- CALM (Stress Management) --
        # High deviation in temperature or lack of speech interactions increases stress (decreases calm)
        stressors = 0.05 * temp_discomfort + 0.02 * min(5, self.steps_since_speech)
        
        # Soothing factors: Classical music (1), Ambient music (4), Comforting speech (7)
        calm_boost = -0.02 - stressors
        if self.current_music == 1:
            calm_boost += 0.06
        elif self.current_music == 4:
            calm_boost += 0.04
        if action == 7:
            calm_boost += 0.15
            
        self.essential[2] += self.dt * calm_boost
        
        # -- ENGAGEMENT (Mental Stimulation) --
        # Monotony decreases engagement. Restored by interaction and stimulating music
        boredom = -0.04
        engagement_boost = boredom
        if self.current_music in [2, 3]:  # Jazz, Pop
            engagement_boost += 0.08
        if action == 8:  # Stimulating speech
            engagement_boost += 0.18
            
        self.essential[3] += self.dt * engagement_boost
        
        # 4. Advance Time of Day
        self.time_of_day = (self.time_of_day + self.dt) % 1.0
        
        # 5. Safe clipping of essential variables
        self.essential = np.clip(self.essential, 0.0, 1.0)
        
        # 6. Retrieve noisy observations
        obs = self._get_obs()
        
        # 7. Compute homeostatic reward (average of comfort variables)
        reward = float(np.mean(self.essential))
        
        done = False
        info = {
            "true_essential": self.essential.copy(),
            "room_temp": self.room_temp,
            "target_temp": self.target_temp,
            "time_of_day": self.time_of_day,
            "music": self.current_music
        }
        
        return obs, reward, done, info
