import streamlit as st
import json
import subprocess
import os
import time
from pathlib import Path

# Define working directory
CONFIG_PATH = "config.json"

st.set_page_config(page_title="Taxation RL Trainer", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Policy Making Framework: Optimal Tax Policy using Deep Reinforcement Learning</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align: center; line-height: 1.4;'>
        <h3>Project Done by:</h3>
        <b>Archit Subidhi</b>: 1BM21AI026<br>
        <b>Aryaman Sharma</b>: 1BM21AI027<br>
        <b>Ayush Kumar Dubey</b>: 1BM21AI028<br>
        <b>Chetna Mundra</b>: 1BM21AI036<br><br>
        <h3>Guide:</h3>
        <b>Dr. Seemanthini K</b>
    </div>
    """,
    unsafe_allow_html=True
)
# Utility to load or create default config
def load_config():
    return {
        "use_wandb": True,
        "algorithm": "PPO",
        "policy_type": "MlpPolicy",
        "total_steps": 1000000,
        "learning_rate": 0.0003,
        "pop_size": 1000,
        "num_states": 9,
        "inital_taxes_params": [0.01, 0.5, 0.5],
        "consumptions_params": [0.8, 5.0],
        "returns_params": [0.04, 0.2],
        "percentiles": [1.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9],
        "episode_steps": 1000,
        "run_id_for_evaluation": "",
        "schedule": "const",
        "n_envs": 4,
        "alpha": 0.8,
        "reward_type": "R1",
        "exp_salaries_params": [10.0, 1.0],
        "wealth_init_params": [12.0, 2.0],
        "action_space_lower_bound": [0.0, 0.0, 0.01],
        "action_space_upper_bound": [0.1, 3.0, 1.0]
    }

# Save config
def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)

# Helper to run scripts
def show_result_graphs(algorithm, run_id):
    if not run_id:
        st.info("No Run ID provided for result plots.")
        return

    result_path = Path("results") / "saved_models" / algorithm / run_id / "sim_plots"
    time.sleep(2)  # small wait to ensure files are saved

    if result_path.exists():
        st.write(f"### \ud83d\udcca Simulation Plots for Run: `{run_id}`")
        for file in sorted(result_path.glob("*.png")):
            st.image(str(file), caption=file.name)
    else:
        st.warning(f"No plots found at `{result_path}`.")

def run_script(script_name, args, mode):
    st.session_state["logs"] = ""
    save_config(config)  # save config before starting

    script_path = Path(script_name)
    if not script_path.exists():
        return  # Simply do nothing if script does not exist

    cmd = f"python {script_name} -f {CONFIG_PATH} {args}"
    st.write(f"### Running `{cmd}`")

    with st.spinner(f"{mode} in progress..."):
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        log_box = st.empty()  # Placeholder for dynamic log updates
        all_logs = ""

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                all_logs += line
                log_box.code(all_logs)  # Show all logs without limit

    st.success(f"{mode} completed.")
    st.code(all_logs, language="bash")  # Display all collected logs

    # After run, display result graphs
    show_result_graphs(config["algorithm"], config.get("run_id_for_evaluation") or config.get("run_id_resume_training"))


# Tabs: Config, Train, Simulate
config_tab, train_tab, simulate_tab = st.tabs(["⚙️ Config", "🚀 Train", "🎮 Simulate"])

# Config Tab
with config_tab:
    st.header("Configuration")
    config = load_config()

    config["algorithm"] = st.selectbox("Algorithm", ["PPO", "A2C", "DQN"], index=["PPO", "A2C", "DQN"].index(config["algorithm"]))
    config["policy_type"] = st.selectbox("Policy Type", ["MlpPolicy"], index=0)
    config["total_steps"] = st.number_input("Total Training Steps", value=config["total_steps"], step=10000)
    config["learning_rate"] = st.number_input("Learning Rate", value=config["learning_rate"], step=0.0001, format="%.4f")
    config["n_envs"] = st.slider("Number of Environments", 1, 16, value=config["n_envs"])
    config["alpha"] = st.slider("Alpha", 0.0, 1.0, value=config["alpha"], step=0.01)
    config["reward_type"] = st.selectbox("Reward Type", ["R1", "EFFICIENCY", "EQUALITY", "MIXED"], index=["R1", "EFFICIENCY", "EQUALITY", "MIXED"].index(config["reward_type"]))
    config["schedule"] = st.selectbox("Schedule", ["const", "exponential", "linear"], index=["const", "exponential", "linear"].index(config["schedule"]))
    config["use_wandb"] = st.checkbox("Use Weights & Biases", value=config["use_wandb"])

    config["run_id_for_evaluation"] = st.text_input("Run ID for Evaluation", value=config["run_id_for_evaluation"])

    st.subheader("Taxation Environment Parameters")
    config["pop_size"] = st.number_input("Population Size", value=config["pop_size"])
    config["num_states"] = st.number_input("Number of States", value=config["num_states"])
    config["episode_steps"] = st.number_input("Episode Steps", value=config["episode_steps"])

    config["percentiles"] = st.text_input("Percentiles (comma-separated)", value=','.join(map(str, config["percentiles"])))
    config["percentiles"] = [float(x.strip()) for x in config["percentiles"].split(",")]

    def list_input(name, default):
        return [float(x.strip()) for x in st.text_input(name, value=','.join(map(str, default))).split(',')]

    config["inital_taxes_params"] = list_input("Initial Taxes Params", config["inital_taxes_params"])
    config["consumptions_params"] = list_input("Consumption Params", config["consumptions_params"])
    config["returns_params"] = list_input("Returns Params", config["returns_params"])
    config["exp_salaries_params"] = list_input("Expected Salaries Params", config["exp_salaries_params"])
    config["wealth_init_params"] = list_input("Wealth Init Params", config["wealth_init_params"])
    config["action_space_lower_bound"] = list_input("Action Space Lower Bound", config["action_space_lower_bound"])
    config["action_space_upper_bound"] = list_input("Action Space Upper Bound", config["action_space_upper_bound"])

    if st.button("💾 Save Config"):
        save_config(config)
        st.success("Configuration saved!")

# Train Tab
with train_tab:
    st.header("Start Training")
    if st.button("🚀 Start Training"):
        run_script("main.py", "", "Training")

# Simulate Tab
with simulate_tab:
    st.header("Run Simulation")
    run_id = st.text_input("Run ID", value="")

    if st.button("🎮 Run Simulation"):
        args = f"--run_id {run_id}"
        run_script("simulate.py", args, "Simulation")

        sim_dir = Path("results") / "saved_models" / config["algorithm"] / run_id / "sim_plots"
        time.sleep(2)
        if sim_dir.exists():
            for file in sim_dir.glob("*.png"):
                st.image(str(file), caption=file.name)
