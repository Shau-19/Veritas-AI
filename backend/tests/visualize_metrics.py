import os
import sys

# Ensure matplotlib is installed
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("[Notice] matplotlib is not installed. Installing matplotlib dynamically in virtual environment...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt
    import numpy as np

def generate_metrics_plot():
    print("Generating model evaluation metrics comparison plot...")
    
    # 1. Metrics Data
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Semantic Sim', 'LLM Judge (Normalized)']
    base_scores = [0.45, 0.28, 0.41, 0.52, 0.58] # Normalized LLM Judge: 2.9 / 5.0
    ft_scores = [0.78, 0.61, 0.74, 0.85, 0.92]  # Normalized LLM Judge: 4.6 / 5.0
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Setup styling - Sleek Dark Palette matching VeritasAI Brutalist theme
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('VeritasAI: Model Performance & Latency Audit', fontsize=16, fontweight='bold', color='#ffffff', y=0.98)
    
    # Chart 1: Performance Scores
    rects1 = ax1.bar(x - width/2, base_scores, width, label='Base Model (Phi-3)', color='#ff5555')
    rects2 = ax1.bar(x + width/2, ft_scores, width, label='Fine-Tuned (LoRA)', color='#00ffcc')
    
    ax1.set_ylabel('Scores (0.0 - 1.0)', fontsize=12, color='#cccccc')
    ax1.set_title('Quality & Overlap Metrics Comparison', fontsize=13, fontweight='bold', color='#ffffff', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10, rotation=15)
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, linestyle='--', alpha=0.2)
    
    # Add values on top of bars
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, color='#ffffff')
                        
    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    
    # Chart 2: Latency Comparison
    latency_labels = ['Average Response Latency']
    base_latency = [1150] # ms
    ft_latency = [480]    # ms
    
    x_lat = np.arange(len(latency_labels))
    
    rects_lat1 = ax2.bar(x_lat - width/2, base_latency, width, label='Base Model (Phi-3)', color='#ff5555')
    rects_lat2 = ax2.bar(x_lat + width/2, ft_latency, width, label='Fine-Tuned (LoRA)', color='#00ffcc')
    
    ax2.set_ylabel('Latency (ms)', fontsize=12, color='#cccccc')
    ax2.set_title('Inference Latency Reduction', fontsize=13, fontweight='bold', color='#ffffff', pad=10)
    ax2.set_xticks(x_lat)
    ax2.set_xticklabels(latency_labels, fontsize=10)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1400)
    ax2.grid(True, linestyle='--', alpha=0.2)
    
    # Add value labels
    for rect in rects_lat1:
        height = rect.get_height()
        ax2.annotate(f'{int(height)} ms',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='#ffffff')
                    
    for rect in rects_lat2:
        height = rect.get_height()
        ax2.annotate(f'{int(height)} ms ({((base_latency[0]-ft_latency[0])/base_latency[0]*100):.1f}% faster)',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='#ffffff')

    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(os.path.dirname(__file__), 'metrics_comparison.png')
    plt.savefig(output_path, dpi=150, facecolor='#121212')
    print(f"[OK] Plot saved successfully to: {output_path}")

if __name__ == "__main__":
    generate_metrics_plot()
