import pandas as pd
import numpy as np
from numpy import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from scipy.stats import norm
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import (brier_score_loss, accuracy_score, recall_score, 
                             f1_score, precision_score, log_loss)

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# Compute Shannon Entropy
def Shannon_Entropy(X, X_sample, Y_sample, Y_train_prior, Y_test_prior):
    kernel = ConstantKernel() * RBF(length_scale_bounds=(.02, 1)) + WhiteKernel()
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, normalize_y=True,
                                     optimizer='fmin_l_bfgs_b', random_state=30)

    # Standardize features
    sclr_X = preprocessing.StandardScaler()
    X_sample_scaled = X_sample # Not scaled  
    model.fit(X_sample_scaled, Y_sample - Y_train_prior)

    # Calculate predictions
    X_scaled = X
    mean, std = model.predict(X_scaled, return_std=True)
    mean += Y_test_prior  # Add prior

    # Calculate the probability that YS > 100 MPa
    threshold = 100
    prob_pos_class = 1 - norm.cdf(threshold, loc=mean, scale=std)

    # Prevent log(0) or log(1) errors
    epsilon = 1e-10
    prob_pos_class = np.clip(prob_pos_class, epsilon, 1 - epsilon)

    # Compute Shannon entropy (binary)
    entropy_values = -prob_pos_class * np.log(prob_pos_class) - (1 - prob_pos_class) * np.log(1 - prob_pos_class)

    return entropy_values, mean, std, prob_pos_class


random.seed(68)

# Initialize lists to store metrics
entropy_lists = [[] for _ in range(15)]
brier_loss_lists = [[] for _ in range(15)]
accuracy_lists = [[] for _ in range(15)]
recall_lists = [[] for _ in range(15)]
f1_lists = [[] for _ in range(15)]
precision_lists = [[] for _ in range(15)]
log_lists = [[] for _ in range(15)]

# 200 Initializations
for j in range(200): #change back to 200
    df = pd.read_csv('/data/CS3_NbTaW_Dataset.csv')
    df['YS 25C PRIOR'] = 0  # No Prior
    noise = np.random.normal(loc=0, scale=0.1 * df['YS T C PRIOR'].std(), size=len(df))
    df['YS T C PRIOR'] += noise  # Add noise to true values

    # True class
    df['YS Greater than 100 MPa'] = np.where(df['YS T C PRIOR'] > 100, 1, 0) 

    # Initialize columns (reset for each initialization)
    elements = ['Nb', 'Ta', 'W']
    df['QV'] = np.nan
    df['Entropy'] = np.nan

    # Randomly sample the initial query value 
    initial_rows = df[df['QV'].isna()].sample(n=1, replace=True).copy()
    initial_rows['QV'] = initial_rows['YS T C PRIOR']
    df.loc[initial_rows.index, 'QV'] = initial_rows['QV']

    # Run the active learning scheme
    df['Entropy'], df['mean'], df['std'], df['Probability_YS_greater_than_100_MPa'] = Shannon_Entropy(
        X=df[elements],
        X_sample=df[elements].loc[df['QV'].notna()],
        Y_sample=df['YS T C PRIOR'].loc[df['QV'].notna()],
        Y_train_prior=df['YS 25C PRIOR'].loc[df['QV'].notna()],
        Y_test_prior=df['YS 25C PRIOR']
    )
    
    # Avoid negative mean values
    df.loc[df['mean'] < 0, 'mean'] = 0
    
    # Predicted class
    df['Pred Class'] = np.where(df['Probability_YS_greater_than_100_MPa'] > 0.5, 1, 0)

    # Compute metrics for values that have not been queried yet 
    brier = brier_score_loss(df['YS Greater than 100 MPa'].loc[df['QV'].isna()],
                             df['Probability_YS_greater_than_100_MPa'].loc[df['QV'].isna()])

    accuracy = accuracy_score(df['YS Greater than 100 MPa'].loc[df['QV'].isna()],
                              df['Pred Class'].loc[df['QV'].isna()])

    recall = recall_score(df['YS Greater than 100 MPa'].loc[df['QV'].isna()],
                          df['Pred Class'].loc[df['QV'].isna()], pos_label=1)

    f1 = f1_score(df['YS Greater than 100 MPa'].loc[df['QV'].isna()],
                  df['Pred Class'].loc[df['QV'].isna()], pos_label=1)

    precision = precision_score(df['YS Greater than 100 MPa'].loc[df['QV'].isna()],
                                df['Pred Class'].loc[df['QV'].isna()], pos_label=1)

    log = log_loss(df['YS Greater than 100 MPa'].loc[df['QV'].isna()],
                   df['Probability_YS_greater_than_100_MPa'].loc[df['QV'].isna()])

    best_index = df[df['QV'].isna()]['Entropy'].idxmax()

    # Store metrics
    entropy_lists[0].append(df['Entropy'].loc[best_index])
    brier_loss_lists[0].append(brier)
    accuracy_lists[0].append(accuracy)
    recall_lists[0].append(recall)
    f1_lists[0].append(f1)
    precision_lists[0].append(precision)
    log_lists[0].append(log)

    # Plot Mean (Predicted Yield Strength)
    #plt.clf()
    #fig, ax = plt.subplots(facecolor="w")
    #plt.figure(figsize=(6, 6))
    #td = TernaryDiagram(materials=df[['Nb', 'Ta', 'W']])
    #td.contour(df[['Nb', 'Ta', 'W']], z=df['mean'], cmap='RdYlGn')
    #td.scatter(df[['Nb', 'Ta', 'W']].loc[df['QV'].notna()], color='lightblue', edgecolor='k', s=150, vmin=0, vmax=270)
    #plt.title('Predicted 1300C Yield Strength', fontsize=14)
    #plt.savefig(f'5_itr_{0}_obj_prior.png', dpi=600, transparent=False, bbox_inches='tight')
    #plt.cla()

    # Iterate over the budget to perform the experiment
    budget = 14
    for itr in range(budget):
        print(f'Beginning ITR {itr}')

        df['Pred Class'] = np.where(df['Probability_YS_greater_than_100_MPa'] > 0.5, 1, 0)

        # Select the best candidate (highest entropy)
        df = df.sample(frac=1)
        best_index = df[df['QV'].isna()]['Entropy'].idxmax()

        # Update QV for the selected best index
        df.loc[best_index, 'QV'] = df.loc[best_index, 'YS T C PRIOR']

       # Update GP after adding the new query value and recalculate probabilities
       # Compute entropy to identify the next point to query
        df['Entropy'], df['mean'], df['std'], df['Probability_YS_greater_than_100_MPa'] = Shannon_Entropy(
            X=df[elements],
            X_sample=df[elements].loc[df['QV'].notna()],
            Y_sample=df['YS T C PRIOR'].loc[df['QV'].notna()],
            Y_train_prior=df['YS 25C PRIOR'].loc[df['QV'].notna()],
            Y_test_prior=df['YS 25C PRIOR']
        )

        # Compute metrics for features that have not been queried yet
        brier = brier_score_loss(df['YS Greater than 100 MPa'].loc[df['QV'].isna()],
                                 df['Probability_YS_greater_than_100_MPa'].loc[df['QV'].isna()])
        accuracy = accuracy_score(df['YS Greater than 100 MPa'].loc[df['QV'].isna()],
                                  df['Pred Class'].loc[df['QV'].isna()])
        recall = recall_score(df['YS Greater than 100 MPa'].loc[df['QV'].isna()],
                              df['Pred Class'].loc[df['QV'].isna()], pos_label=1)
        f1 = f1_score(df['YS Greater than 100 MPa'].loc[df['QV'].isna()],
                      df['Pred Class'].loc[df['QV'].isna()], pos_label=1)
        precision = precision_score(df['YS Greater than 100 MPa'].loc[df['QV'].isna()],
                                    df['Pred Class'].loc[df['QV'].isna()], pos_label=1)
        log = log_loss(df['YS Greater than 100 MPa'].loc[df['QV'].isna()],
                       df['Probability_YS_greater_than_100_MPa'].loc[df['QV'].isna()])

        # Store metrics
        entropy_lists[itr+1].append(df['Entropy'].loc[best_index])
        brier_loss_lists[itr+1].append(brier)
        accuracy_lists[itr+1].append(accuracy)
        recall_lists[itr+1].append(recall)
        f1_lists[itr+1].append(f1)
        precision_lists[itr+1].append(precision)
        log_lists[itr+1].append(log)

        # Avoid negative mean values
        df.loc[df['mean'] < 0, 'mean'] = 0

        # Plot Mean (Predicted Yield Strength)
        #plt.clf()
        #fig, ax = plt.subplots(facecolor="w")
        #plt.figure(figsize=(6, 6))
        #td = TernaryDiagram(materials=df[['Nb', 'Ta', 'W']])
        #td.contour(df[['Nb', 'Ta', 'W']], z=df['mean'], cmap='RdYlGn')
        #td.scatter(df[['Nb', 'Ta', 'W']].loc[df['QV'].notna()], color='lightblue', edgecolor='k', s=150, vmin=0, vmax=270)
        #plt.title('Predicted 1300C Yield Strength', fontsize=14)
        #plt.savefig(f'5_itr_{itr+1}_obj_prior.png', dpi=600, transparent=True, bbox_inches='tight')
        #plt.cla()
        
# Initialize dictionaries to store the mean and standard deviation for each metric
metrics = {
    'Brier Loss': {'mean': [], 'std': []},
    'Accuracy': {'mean': [], 'std': []},
    'Recall': {'mean': [], 'std': []},
    'F1 Score': {'mean': [], 'std': []},
    'Precision': {'mean': [], 'std': []},
    'Log Loss': {'mean': [], 'std': []},
}

# Store the mean and standard deviation of each metric per iteration across 200 initializations
for brier_loss_list, accuracy_list, recall_list, f1_list, precision_list, log_list in zip(brier_loss_lists, accuracy_lists, recall_lists, f1_lists, precision_lists, log_lists):
    
    # Update metrics for each list
    metrics['Brier Loss']['mean'].append(np.mean(brier_loss_list))
    metrics['Brier Loss']['std'].append(np.std(brier_loss_list))
    
    metrics['Accuracy']['mean'].append(np.mean(accuracy_list))
    metrics['Accuracy']['std'].append(np.std(accuracy_list))
    
    metrics['Recall']['mean'].append(np.mean(recall_list))
    metrics['Recall']['std'].append(np.std(recall_list))
    
    metrics['F1 Score']['mean'].append(np.mean(f1_list))
    metrics['F1 Score']['std'].append(np.std(f1_list))
    
    metrics['Precision']['mean'].append(np.mean(precision_list))
    metrics['Precision']['std'].append(np.std(precision_list))
    
    metrics['Log Loss']['mean'].append(np.mean(log_list))
    metrics['Log Loss']['std'].append(np.std(log_list))

# Create DataFrame with results; save to an Excel file
results = pd.DataFrame({
    'Brier Loss Mean': metrics['Brier Loss']['mean'],
    'Brier Loss Std': metrics['Brier Loss']['std'],
    'Accuracy Mean': metrics['Accuracy']['mean'],
    'Accuracy Std': metrics['Accuracy']['std'],
    'Recall Mean': metrics['Recall']['mean'],
    'Recall Std': metrics['Recall']['std'],
    'F1 Score Mean': metrics['F1 Score']['mean'],
    'F1 Score Std': metrics['F1 Score']['std'],
    'Precision Mean': metrics['Precision']['mean'],
    'Precision Std': metrics['Precision']['std'],
    'Log Loss Mean': metrics['Log Loss']['mean'],
    'Log Loss Std': metrics['Log Loss']['std'],
})

results.to_excel('/data/ys_avg_metrics_no_prior.xlsx', index=False)
