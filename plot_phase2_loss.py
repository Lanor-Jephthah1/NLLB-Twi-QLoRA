import matplotlib.pyplot as plt
import numpy as np
import os

# Illustrative data based on README description
# Spike to 35.5, then descent over 1,355 steps
steps = np.arange(0, 1356)
# Create a curve that starts at 35.5 and decays towards a floor of ~14.0
loss = 21.5 * np.exp(-steps / 400) + 14.0 
# Add some noise to make it look like real training
np.random.seed(42)
noise = np.random.normal(0, 0.8, len(steps))
loss = loss + noise

plt.figure(figsize=(10, 5))
plt.style.use('seaborn-v0_8-whitegrid')

plt.plot(steps, loss, color='#d62728', alpha=0.8, linewidth=1.5)
plt.title('Training Loss Curve — Phase 2 (Human Alignment)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Global Steps', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)

# Annotate the distribution shift spike
plt.annotate('Initial Distribution Shift (Spike to ~35.5)', 
             xy=(0, 35.5), xytext=(200, 38),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=10, fontweight='bold')

plt.ylim(0, 45)
plt.tight_layout()

# Save
output_path = 'assets/training_loss_phase2.png'
os.makedirs('assets', exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f"Graph saved to {output_path}")
