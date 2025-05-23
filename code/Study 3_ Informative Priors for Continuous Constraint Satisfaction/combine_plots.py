import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from Excel files
df = pd.read_excel('/data/ys_avg_metrics_no_prior.xlsx')
df_prior = pd.read_excel('/data/ys_avg_metrics_prior.xlsx')


iterations = np.arange(1, 16)

# Plot function 
def plot_error_bar(ax, iterations, data, data_prior, y_label, metric):
    ax.fill_between(iterations, data[f'{metric} Mean'] - data[f'{metric} Std'], 
                    data[f'{metric} Mean'] + data[f'{metric} Std'], color='red', alpha=0.3)
    ax.fill_between(iterations, data_prior[f'{metric} Mean (Prior)'] - data_prior[f'{metric} Std (Prior)'], 
                    data_prior[f'{metric} Mean (Prior)'] + data_prior[f'{metric} Std (Prior)'], color='blue', alpha=0.3)

    ax.plot(iterations, data[f'{metric} Mean'], label='No Prior', color='red', lw=2)
    ax.plot(iterations, data_prior[f'{metric} Mean (Prior)'], label='Prior', color='blue', lw=2)

    #ax.set_title(f'{metric} vs Iteration')
    ax.set_title(f'{metric}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'{metric} (Mean ± Std)')
    ax.set_xlim(1, 15)
    ax.legend()

# Create a 2 by 3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

fig.suptitle('Error Metrics Across Iterations', fontsize=16)

# List of metrics 
metrics = [ 'Accuracy', 'Precision', 'Brier Loss','Recall', 'F1 Score', 'Log Loss']

axs = axs.flatten()

# Plot each metric in the corresponding subplot
for i, metric in enumerate(metrics):
    plot_error_bar(axs[i], iterations, df, df_prior, metric, metric)

plt.tight_layout()
plt.savefig('/results/Continuous_Constraint_Metrics.png', dpi=600, bbox_inches='tight')
plt.show()
