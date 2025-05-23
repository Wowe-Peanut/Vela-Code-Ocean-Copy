import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df_prior = pd.read_csv('/data/Error_Metric_w_Prior.csv')
df_noprior = pd.read_csv('/data/Error_Metric_wo_Prior.csv')


# Filter data based on iteration threshold
df_prior = df_prior[df_prior['ITR'] <= 30]
df_noprior = df_noprior[df_noprior['ITR'] <= 30]

# Identify metrics to compute
metrics = ['Brier_Loss', 'Log_Loss', 'Accuracy', 'Precision','Recall','F1']
indices = range(0,200)  # Adjust this if the range changes, 2

# Loop through metrics and indices to compute statistics
for metric in metrics:
    # Collect the columns related to the current metric
    columns_prior = [f"{metric}_{i}" for i in indices]
    columns_noprior = [f"{metric}_{i}" for i in indices]

    # Compute mean and std across the selected columns grouped by ITR
    df_prior_stats = df_prior.groupby('ITR')[columns_prior].mean().agg(['mean', 'std'], axis=1).reset_index()
    df_noprior_stats = df_noprior.groupby('ITR')[columns_noprior].mean().agg(['mean', 'std'], axis=1).reset_index()

    # Plot mean and standard deviation with shaded regions
    plt.figure(figsize=(4, 4))

    # Plot for Prior with shaded region
    plt.fill_between(df_prior_stats['ITR'],
                     df_prior_stats['mean'] - df_prior_stats['std'],
                     df_prior_stats['mean'] + df_prior_stats['std'],
                     color='blue', alpha=0.3, )

    # Plot for No Prior with shaded region
    plt.fill_between(df_noprior_stats['ITR'],
                     df_noprior_stats['mean'] - df_noprior_stats['std'],
                     df_noprior_stats['mean'] + df_noprior_stats['std'],
                     color='red', alpha=0.3, )

    # Plot the mean values (lines) for Prior and No Prior
    plt.plot(df_prior_stats['ITR'], df_prior_stats['mean'], label='Prior', color='blue', lw=2)
    plt.plot(df_noprior_stats['ITR'], df_noprior_stats['mean'], label='No Prior', color='red', lw=2)

    # Labeling the plot
    plt.xlabel('Iteration (ITR)')
    plt.ylabel(f'{metric} (Mean ± Std)')
    plt.title(f'{metric} vs Iteration')
    plt.legend()
    plt.grid()
    plt.xlim(0,25)
    plt.savefig(f'/results/{metric}_vs_Iteration_shaded.png', dpi=600,bbox_inches='tight')
    plt.show()
