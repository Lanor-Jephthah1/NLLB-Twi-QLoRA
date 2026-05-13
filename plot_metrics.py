import matplotlib.pyplot as plt
import os

# Data from README
labels = ['Baseline', 'CP 3.5k', 'CP 7.5k', 'Phase 1 Final', 'Phase 2 (Final)']
bleu_scores = [18.94, 42.87, 43.42, 43.37, 41.99]
chrf_scores = [0, 62.22, 63.05, 63.16, 61.21] # Baseline chrF++ was not measured

# Create plot
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot BLEU
color_bleu = '#1f77b4'
line1, = plt.plot(labels, bleu_scores, marker='o', linestyle='-', linewidth=2.5, color=color_bleu, label='BLEU Score')
for i, txt in enumerate(bleu_scores):
    plt.annotate(f'{txt:.2f}', (labels[i], bleu_scores[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color=color_bleu)

# Plot chrF++ (on the same scale or separate? They are both 0-100, so same is fine)
color_chrf = '#ff7f0e'
line2, = plt.plot(labels[1:], chrf_scores[1:], marker='s', linestyle='--', linewidth=2, color=color_chrf, label='chrF++ Score')
for i, txt in enumerate(chrf_scores[1:]):
    plt.annotate(f'{txt:.2f}', (labels[i+1], chrf_scores[i+1]), textcoords="offset points", xytext=(0,-15), ha='center', fontweight='bold', color=color_chrf)

plt.title('NLLB-Twi Translation Performance Progression', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Training Phase / Checkpoint', fontsize=12)
plt.ylabel('Score (0-100)', fontsize=12)
plt.ylim(0, 75) # Set range to show growth from baseline
plt.legend(handles=[line1, line2], loc='lower right', frameon=True)

# Save
output_path = 'assets/evaluation_metrics.png'
os.makedirs('assets', exist_ok=True)
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"Graph saved to {output_path}")
