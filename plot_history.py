import json
import matplotlib.pyplot as plt
import os

# Path to your trainer state (Using the latest 12000 checkpoint)
JSON_PATH = "f:/twi_translation_model/checkpoint-12000/trainer_state.json"
SAVE_PATH = "C:/Users/McLanor Jeff/.gemini/antigravity/scratch/training_loss_clean.png"

def plot_json_history():
    if not os.path.exists(JSON_PATH):
        print(f"Error: Could not find {JSON_PATH}")
        return

    print(f"Loading history from {JSON_PATH}...")
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    # Extract log history
    history = data.get("log_history", [])
    
    steps = []
    losses = []
    
    for entry in history:
        if "loss" in entry and "step" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])

    if not steps:
        print("No loss data found in the JSON!")
        return

    # Create Plot
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, color='#2ecc71', linewidth=1, label='Training Loss')
    
    plt.title("Twi-English Translation: Training Loss (Scaling Phase)", fontsize=14, fontweight='bold')
    plt.xlabel("Global Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Save the professional version
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300)
    print(f"✅ SUCCESS! Professional graph saved to: {SAVE_PATH}")
    plt.show()

if __name__ == "__main__":
    plot_json_history()
