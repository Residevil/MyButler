# 👤 MyButler - Biological Homeostatic AI Butler

MyButler is a prototype AI companion built on first principles of biological intelligence. Unlike traditional disembodied AI systems trained offline on static datasets, MyButler behaves like a living organism. It acts continuously in a closed sensorimotor loop, driven by two primordial motives: **survival** (relational homeostasis of its human's well-being) and **curiosity** (uncertainty reduction over its world model).

---

## 🧬 Scientific & Theoretical Foundations

MyButler is modeled after **Active Inference** and the **Free Energy Principle**, a leading framework in theoretical neuroscience pioneered by Dr. Karl Friston. 

```
                                  [ Environment (Human + Home) ]
                                            /        \
                                  Observations      Actions
                                          /            \
                       [ Generative Model ] <-------- [ Policy Selection ]
                        (Predicts Comfort,             (Minimizes Future
                         Calm, Engagement)              Free Energy)
```

### Core Architecture:
1. **Valence & Need**: Every input and state has valence. The Butler has an interoceptive drive to maintain the human's Comfort, Calm, Health, and Engagement. Any deviation from the optimal range generates interoceptive prediction error—interpreted as biological "need" or discomfort.
2. **Generative Model**: A learned dynamic neural model consisting of:
   * **Transition Model**: Predicts the next state based on the current state and chosen action.
   * **Observation Model**: Map hidden states to expected sensory observations.
   * **Interoceptive Model**: Predicts human homeostatic variables.
3. **Active Inference & Planning**: Instead of training on hand-coded rewards, the Butler acts to minimize its future **Expected Free Energy (EFE)**:
   * **Survival Drive**: Minimize deviations of the predicted interoceptive state from preferred, high-evidence well-being targets.
   * **Curiosity Drive (Epistemic Value)**: Actively seek out novel/unfamiliar situations (e.g., trying a different music genre or conversation style) to reduce uncertainty in its world model.
4. **Identity Prior**: A fixed top-down axiom in the generative model that locks its self-identity: *"I am an artificial butler dedicated to assisting and maintaining my human's well-being while respecting their autonomy."* This serves as an embedded, unalterable safety/ethical guardrail.

---

## 📂 Repository Structure

```
├── butler/
│   ├── env.py            # Homeostatic environment simulation (comfort, calm, music, heat)
│   ├── models.py         # PyTorch generative models (Transition, Observation, Interoception)
│   ├── agent.py          # Active Inference agent (belief updating, free energy, planning)
│   └── simulation.py     # Simulation orchestrator and visual state history generator
├── logs/
│   └── system.log        # Online operations & learning logs
├── tests/
│   └── test_env.py       # Automated unit tests for environment dynamics
├── .gitignore            # Standard Python & PyTorch git exclusion rules
└── requirements.txt      # Dependency definition
```

---

## 🛠️ Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Homeostatic Simulation
To launch the closed-loop active inference simulation and watch MyButler interact with a simulated human:
```bash
python butler/simulation.py
```
This runs a multi-step interactive simulation, tracks variables over time, logs the agent's decisions to `logs/system.log`, and generates a visual performance breakdown saved in the workspace.

---

## 📈 Developmental Phases

*   **Phase 0: The Medium & The Body** (Simulated homeostatic comfort loop) - *Current Phase*
*   **Phase 1: Reflexes & Hardwired Valence**
*   **Phase 2: Local Plasticity & Conditioning** (Hebbian online updates without backprop)
*   **Phase 3: Active Inference Hierarchy**
*   **Phase 4: Curiosity-Driven Exploration**
*   **Phase 5: Imitation (Copying) and Social Dynamics**
