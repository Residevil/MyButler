import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np
import matplotlib.pyplot as plt
from butler.env import ButlerEnv
from butler.agent import ActiveInferenceAgent

def setup_logging():
    """Sets up file-based logging for tracking system operations and agent learning."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "system.log")
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("MyButler")

def action_to_string(action):
    """Translates integer actions to human-readable strings."""
    mapping = {
        0: "Do nothing",
        1: "Thermostat setpoint +1°C",
        2: "Thermostat setpoint -1°C",
        3: "Play Classical music",
        4: "Play Jazz music",
        5: "Play Pop music",
        6: "Play Ambient music",
        7: "Speak comforting phrase",
        8: "Speak stimulating phrase"
    }
    return mapping.get(action, "Unknown Action")

DIAGNOSTIC_FIELDS = [
    "belief_calm_raw",
    "belief_calm_pinned",
    "predicted_calm_obs_pre",
    "predicted_calm_obs",
    "predicted_calm_intero",
    "obs_calm",
    "obs_error_calm_vs_obs",
    "obs_error_calm_vs_true",
    "intero_error_calm_vs_true",
    "belief_calm_delta",
    "sensory_error_calm",
    "transition_pred_calm",
    "transition_error_calm",
    "transition_logvar_calm",
]


def run_simulation(steps=150, diagnostic_csv=None):
    logger = setup_logging()
    logger.info("Initializing MyButler Homeostatic Active Inference Simulation")
    
    # Initialize Environment and Agent
    env = ButlerEnv()
    agent = ActiveInferenceAgent()
    
    # Reset env and agent
    obs = env.reset()
    agent.reset(obs)
    
    # Histories for plotting
    history = {
        "true_comfort": [],
        "true_health": [],
        "true_calm": [],
        "true_engagement": [],
        
        "belief_comfort": [],
        "belief_health": [],
        "belief_calm": [],
        "belief_engagement": [],
        
        "room_temp": [],
        "target_temp": [],
        "time_of_day": [],
        "music": [],
        "actions": [],
        
        "loss_total": [],
        "loss_trans": [],
        "loss_obs": [],
        "loss_essential": [],
        **{k: [] for k in DIAGNOSTIC_FIELDS},
    }

    diag_rows = []
    
    logger.info("Starting closed sensorimotor loop execution...")
    
    last_action = None
    prev_transition_pred_calm = None
    for step in range(steps):
        # 1. Belief update (Sensory correction)
        belief, diag = agent.update_belief(obs, last_action)
        
        # 2. Action planning & selection
        action, (survival_cost, epistemic_val, efe) = agent.select_action()
        
        # 3. Environment Step
        next_obs, reward, done, info = env.step(action)
        
        # Extract essential variables directly from info
        true_e = info["true_essential"]
        true_calm = float(true_e[2])
        diag["true_calm"] = true_calm
        diag["obs_error_calm_vs_true"] = diag["predicted_calm_obs"] - true_calm
        diag["intero_error_calm_vs_true"] = diag["predicted_calm_intero"] - true_calm
        diag["transition_error_calm"] = (
            None if prev_transition_pred_calm is None
            else prev_transition_pred_calm - true_calm
        )
        prev_transition_pred_calm = diag["transition_pred_calm"]
        
        # 4. Store experience in buffer
        agent.store_transition(belief, action, next_obs, next_obs, true_e)
        
        # 5. Online model learning update step
        losses = agent.learn()
        
        # Logging current status
        logger.info(
            f"Step {step:03d} | Time: {info['time_of_day']:.2f} | "
            f"Act: {action_to_string(action)} | "
            f"Calm: {true_calm:.2f} (Belief: {belief[2]:.2f}) | "
            f"pred_obs={diag['predicted_calm_obs']:.3f} "
            f"pred_intero={diag['predicted_calm_intero']:.3f} | "
            f"err_true={diag['obs_error_calm_vs_true']:+.3f} "
            f"intero_gap={diag['intero_error_calm_vs_true']:+.3f} | "
            f"Δbelief={diag['belief_calm_delta']:+.4f} "
            f"pinned={int(diag['belief_calm_pinned'])} | "
            f"EFE: {efe:.3f}"
        )
        
        # Record histories
        history["true_comfort"].append(true_e[0])
        history["true_health"].append(true_e[1])
        history["true_calm"].append(true_e[2])
        history["true_engagement"].append(true_e[3])
        
        history["belief_comfort"].append(belief[0])
        history["belief_health"].append(belief[1])
        history["belief_calm"].append(belief[2])
        history["belief_engagement"].append(belief[3])
        
        history["room_temp"].append(info["room_temp"])
        history["target_temp"].append(info["target_temp"])
        history["time_of_day"].append(info["time_of_day"])
        history["music"].append(info["music"])
        history["actions"].append(action)
        
        if losses:
            history["loss_total"].append(losses["total"])
            history["loss_trans"].append(losses["transition"])
            history["loss_obs"].append(losses["obs"])
            history["loss_essential"].append(losses["essential"])
        else:
            history["loss_total"].append(0.0)
            history["loss_trans"].append(0.0)
            history["loss_obs"].append(0.0)
            history["loss_essential"].append(0.0)

        for key in DIAGNOSTIC_FIELDS:
            history[key].append(diag.get(key))
        diag_rows.append({
            "step": step,
            "true_calm": true_calm,
            "belief_calm": diag["belief_calm"],
            **{k: diag.get(k) for k in DIAGNOSTIC_FIELDS},
        })
            
        # Update trackers
        obs = next_obs
        last_action = action
        
    logger.info("Simulation execution complete! Saving results and generating visualizations...")

    if diagnostic_csv:
        _write_diagnostic_csv(diagnostic_csv, diag_rows)
        logger.info("Inference diagnostics saved to %s", diagnostic_csv)

    _log_diagnostic_summary(logger, history, steps)
    
    # 6. Render Premium Visualization
    plot_results(history)
    logger.info("Visualization saved to logs/butler_performance.png")


def _write_diagnostic_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "true_calm", "belief_calm"] + DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _log_diagnostic_summary(logger, history, steps):
    """High-signal summary: obs model vs intero model vs belief update."""
    pred_obs = np.array(history["predicted_calm_obs"], dtype=float)
    pred_intero = np.array(history["predicted_calm_intero"], dtype=float)
    err_true = np.array(history["obs_error_calm_vs_true"], dtype=float)
    intero_gap = np.array(history["intero_error_calm_vs_true"], dtype=float)
    belief = np.array(history["belief_calm"], dtype=float)
    true_calm = np.array(history["true_calm"], dtype=float)
    delta = np.array(history["belief_calm_delta"], dtype=float)
    sensory = np.array(history["sensory_error_calm"], dtype=float)
    pinned = np.array(history["belief_calm_pinned"], dtype=float)

    def _phase(name, start, stop):
        if start >= stop:
            return
        sl = slice(start, stop)
        trans_err = history["transition_error_calm"][sl]
        trans_vals = [x for x in trans_err if x is not None]
        trans_mean = np.mean(np.abs(trans_vals)) if trans_vals else float("nan")
        logger.info(
            "%s (steps %d-%d): true=%.3f belief=%.3f |obs_err|=%.3f "
            "|intero_gap|=%.3f |Δbelief|=%.4f sensory=%.5f pinned=%.0f%% "
            "trans_err=%.3f",
            name,
            start,
            stop - 1,
            np.mean(true_calm[sl]),
            np.mean(belief[sl]),
            np.mean(np.abs(err_true[sl])),
            np.mean(np.abs(intero_gap[sl])),
            np.mean(np.abs(delta[sl])),
            np.mean(sensory[sl]),
            100.0 * np.mean(pinned[sl]),
            trans_mean,
        )

    logger.info("--- Calm inference diagnostic summary ---")
    _phase("Early", 0, min(20, steps))
    if steps > 40:
        _phase("Mid (delusion onset ~50)", 40, min(60, steps))
    _phase("Late", max(0, steps - 20), steps)
    logger.info(
        "Full run: pred_obs=%.3f pred_intero=%.3f true=%.3f belief=%.3f | "
        "mean |obs_err|=%.3f mean |intero_gap|=%.3f pinned=%.0f%%",
        pred_obs.mean(),
        pred_intero.mean(),
        true_calm.mean(),
        belief.mean(),
        np.mean(np.abs(err_true)),
        np.mean(np.abs(intero_gap)),
        100.0 * pinned.mean(),
    )
    if np.mean(np.abs(intero_gap)) > np.mean(np.abs(err_true)) + 0.05:
        logger.info(
            "Diagnosis hint: interoceptive head over-predicts Calm more than "
            "observation model — EFE planner likely acting on intero delusion."
        )
    elif np.mean(np.abs(err_true)) > 0.1 and np.mean(np.abs(delta)) < 0.01:
        logger.info(
            "Diagnosis hint: large obs error but tiny belief updates — "
            "inference ignoring sensory evidence or beliefs pinned at clip bounds."
        )

def plot_results(history):
    """Generates a beautiful, publication-ready visualization of the simulation run."""
    steps = len(history["true_comfort"])
    t = np.arange(steps)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(3, 2, figsize=(15, 12), dpi=150)
    
    # Harmonious colors
    c_comfort = '#2ecc71'  # Green
    c_health = '#3498db'   # Blue
    c_calm = '#9b59b6'     # Purple
    c_engagement = '#f1c40f'  # Yellow/Gold
    
    # Panel 1: Comfort & Health (True vs Belief)
    axs[0, 0].plot(t, history["true_comfort"], label="True Comfort", color=c_comfort, linewidth=2)
    axs[0, 0].plot(t, history["belief_comfort"], label="Belief Comfort", color=c_comfort, linestyle="--", alpha=0.8)
    axs[0, 0].plot(t, history["true_health"], label="True Health", color=c_health, linewidth=2)
    axs[0, 0].plot(t, history["belief_health"], label="Belief Health", color=c_health, linestyle="--", alpha=0.8)
    axs[0, 0].set_title("Comfort & Health Dynamics", fontsize=12, fontweight='bold')
    axs[0, 0].set_ylim(-0.05, 1.05)
    axs[0, 0].legend(loc="lower left")
    
    # Panel 2: Calm & Engagement (True vs Belief)
    axs[0, 1].plot(t, history["true_calm"], label="True Calm", color=c_calm, linewidth=2)
    axs[0, 1].plot(t, history["belief_calm"], label="Belief Calm", color=c_calm, linestyle="--", alpha=0.8)
    axs[0, 1].plot(t, history["true_engagement"], label="True Engagement", color=c_engagement, linewidth=2)
    axs[0, 1].plot(t, history["belief_engagement"], label="Belief Engagement", color=c_engagement, linestyle="--", alpha=0.8)
    axs[0, 1].set_title("Calm & Engagement Dynamics", fontsize=12, fontweight='bold')
    axs[0, 1].set_ylim(-0.05, 1.05)
    axs[0, 1].legend(loc="lower left")
    
    # Panel 3: Physical Environment (Temperature)
    axs[1, 0].plot(t, history["room_temp"], label="Room Temp (°C)", color='#e74c3c', linewidth=2)
    axs[1, 0].plot(t, history["target_temp"], label="Target Temp (°C)", color='#e74c3c', linestyle=":", alpha=0.8)
    axs[1, 0].axhline(22.0, color='gray', linestyle='--', alpha=0.5, label="Optimal (22°C)")
    axs[1, 0].set_title("Home Temperature Control", fontsize=12, fontweight='bold')
    axs[1, 0].legend()
    
    # Panel 4: Actions Selected
    axs[1, 1].scatter(t, history["actions"], c=history["actions"], cmap="plasma", alpha=0.7, edgecolors='none', s=40)
    axs[1, 1].set_yticks(np.arange(9))
    axs[1, 1].set_yticklabels([
        "Do Nothing", "+1°C Heat", "-1°C Cool", "Classical", "Jazz", "Pop", "Ambient", "Comfort Speech", "Stim Speech"
    ], fontsize=8)
    axs[1, 1].set_title("Agent Sensorimotor Interventions (Actions)", fontsize=12, fontweight='bold')
    
    # Panel 5: Average Human Well-being (Reward)
    avg_wellbeing = [np.mean([history["true_comfort"][i], history["true_health"][i], history["true_calm"][i], history["true_engagement"][i]]) for i in range(steps)]
    axs[2, 0].fill_between(t, avg_wellbeing, 0.7, where=(np.array(avg_wellbeing) >= 0.7), color='#2ecc71', alpha=0.3, label="Optimal Zone (>0.7)")
    axs[2, 0].fill_between(t, avg_wellbeing, 0.7, where=(np.array(avg_wellbeing) < 0.7), color='#e74c3c', alpha=0.3, label="Discomfort Zone (<0.7)")
    axs[2, 0].plot(t, avg_wellbeing, color='#2c3e50', linewidth=2, label="Average Well-Being")
    axs[2, 0].set_title("Overall Human Well-Being Over Time", fontsize=12, fontweight='bold')
    axs[2, 0].set_ylim(0.0, 1.05)
    axs[2, 0].legend(loc="lower left")
    
    # Panel 6: Generative Model Online Training Loss
    loss_total = np.array(history["loss_total"])
    # Plot non-zero loss values
    non_zero = loss_total > 0.0
    if np.any(non_zero):
        axs[2, 1].plot(t[non_zero], loss_total[non_zero], color='#f39c12', label="Total Loss", linewidth=1.5)
        axs[2, 1].plot(t[non_zero], np.array(history["loss_trans"])[non_zero], color='#d35400', label="Transition Loss", alpha=0.7)
        axs[2, 1].plot(t[non_zero], np.array(history["loss_obs"])[non_zero], color='#c0392b', label="Obs Loss", alpha=0.7)
        axs[2, 1].plot(t[non_zero], np.array(history["loss_essential"])[non_zero], color='#2980b9', label="Interoceptive Loss", alpha=0.7)
        axs[2, 1].set_yscale('log')
    axs[2, 1].set_title("Online Generative Model Loss (Log Scale)", fontsize=12, fontweight='bold')
    axs[2, 1].legend()
    
    plt.tight_layout()
    
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/butler_performance.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MyButler active inference simulation")
    parser.add_argument("--steps", type=int, default=150, help="Simulation length")
    parser.add_argument(
        "--diagnostic-csv",
        default="logs/inference_diagnostics.csv",
        help="CSV path for per-step calm inference diagnostics",
    )
    args = parser.parse_args()
    run_simulation(steps=args.steps, diagnostic_csv=args.diagnostic_csv)
