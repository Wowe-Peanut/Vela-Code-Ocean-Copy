import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF , WhiteKernel, ConstantKernel, DotProduct
from sklearn.preprocessing import MinMaxScaler
from scipy.special import logit, softmax
import mpltern
from sklearn.metrics import brier_score_loss, log_loss, precision_score, recall_score, accuracy_score, f1_score

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class GaussianProcessWithPrior:
    def __init__(self, kernel=None, normalize_y=True, n_restarts_optimizer=10):
        self.gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=normalize_y, n_restarts_optimizer=n_restarts_optimizer,)#optimizer=None
        self.train_prior = None  # Prior for training data
        self.predict_prior = None  # Prior for prediction data

    def fit(self, X, y, train_prior):
        """Fits the model to the residuals (y - train_prior)."""
        if train_prior.shape != y.shape:
            raise ValueError("The shape of 'train_prior' must match the shape of 'y'")
        self.train_prior = train_prior
        residuals = y - train_prior
        self.gpr.fit(X, residuals)

    def predict(self, X, predict_prior, return_std=True):
        """Predicts using the GP model with the prior added back."""
        if predict_prior.shape[0] != X.shape[0]:
            raise ValueError("The shape of 'predict_prior' must match the number of prediction points in 'X'")

        mu, std = self.gpr.predict(X, return_std=True)
        mu_with_prior = mu + predict_prior

        if return_std:
            return sigmoid(mu_with_prior), std
        return sigmoid(mu_with_prior)

class EnsembleGaussianProcess:
    def __init__(self, kernel=None, normalize_y=False, n_restarts_optimizer=10,return_std=True):
        self.gpc_bcc = GaussianProcessWithPrior(kernel=kernel, normalize_y=normalize_y, n_restarts_optimizer=n_restarts_optimizer,)
        self.gpc_fcc = GaussianProcessWithPrior(kernel=kernel, normalize_y=normalize_y, n_restarts_optimizer=n_restarts_optimizer,)
        self.gpc_mix = GaussianProcessWithPrior(kernel=kernel, normalize_y=normalize_y, n_restarts_optimizer=n_restarts_optimizer,)
        self.return_std = return_std

    def fit(self, X, y_bcc, y_fcc, y_mix, prior_bcc, prior_fcc, prior_mix):
        """Fits the individual models for BCC, FCC, and MIX."""
        self.gpc_bcc.fit(X, y_bcc, prior_bcc)
        self.gpc_fcc.fit(X, y_fcc, prior_fcc)
        self.gpc_mix.fit(X, y_mix, prior_mix)

    def predict(self, X, prior_bcc, prior_fcc, prior_mix):
        """Predicts using the softmax of all three models (BCC, FCC, MIX)."""
        prob_bcc,std_latent_bcc = self.gpc_bcc.predict(X, prior_bcc,return_std=self.return_std)
        prob_fcc,std_latent_fcc = self.gpc_fcc.predict(X, prior_fcc,return_std=self.return_std)
        prob_mix,std_latent_mix = self.gpc_mix.predict(X, prior_mix,return_std=self.return_std)

        # Stack the predictions for BCC, FCC, and MIX
        predictions = np.vstack([prob_bcc/1, prob_fcc/1, prob_mix/1]).T # T = 0.25 is the softmax temperature
        predictions = predictions / predictions.sum(axis=1, keepdims=True)
        print('*********')
        #Naively combine the standard deviations of the latent GPRs
        combined_latent_std = std_latent_bcc + std_latent_fcc + std_latent_mix


        print('**************************')


        # Apply softmax normalization across the predictions
        #probabilities = softmax(predictions, axis=1)
        probabilities = predictions

        # Calculate entropy for the probability distribution
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)

        return probabilities,combined_latent_std, entropy

# Acquisition function based on entropy
def acquisition_function(entropy):
    """Selects the data points with the highest entropy."""
    return np.argsort(entropy)[::-1]

df_error_metrics = pd.DataFrame()

for j in range(200):
    # Load data
    df = pd.read_csv('/data/CS2_FeNiCr_Dataset.csv')

    # Prepare the DataFrame with relevant columns
    scaler = MinMaxScaler(feature_range=(logit(0.4999999999), logit(1-0.4999999999)))
    #scaler = MinMaxScaler(feature_range=(logit(0.3), logit(1-0.3)))

    df.loc[df['VEC mean'] >= 8, 'VEC_condition'] = 0  # FCC
    df.loc[df['VEC mean'] <= 6.87, 'VEC_condition'] = 1  # BCC
    df.loc[(df['VEC mean'] > 6.87) & (df['VEC mean'] < 8), 'VEC_condition'] = 0.5  # MIX

    # Set mutually exclusive priors
    df['Prior_FCC'] = (df['VEC_condition'] == 0).astype(int)
    df['Prior_BCC'] = (df['VEC_condition'] == 1).astype(int)
    df['Prior_MIX'] = (df['VEC_condition'] == 0.5).astype(int)

    # Normalize the priors
    df['Prior_FCC'] = scaler.fit_transform(df['Prior_FCC'].values.reshape(-1, 1))
    df['Prior_BCC'] = scaler.fit_transform(df['Prior_BCC'].values.reshape(-1, 1))
    df['Prior_MIX'] = scaler.fit_transform(df['Prior_MIX'].values.reshape(-1, 1))

    # Normalize the Prior_MIX using the same scaler as for Prior_FCC and Prior_BCC
    df['Prior_MIX'] = scaler.fit_transform(df['Prior_MIX'].values.reshape(-1, 1))

    scaler = MinMaxScaler(feature_range=(-5, 5))
    # Set Truth_BCC to 1 if 'EQ 1000C BCC_B2' is greater than 0.99, otherwise 0
    df['Truth_BCC'] = (df['EQ 1000C BCC_B2'] > 0.99).astype(int)

    # Set Truth_FCC to 1 if 'EQ 1000C FCC_L12' is greater than 0.99, otherwise 0
    df['Truth_FCC'] = (df['EQ 1000C FCC_L12'] > 0.99).astype(int)

    # Set Truth_MIX to 1 if both BCC and FCC phases are present (i.e., neither is a single-phase)
    df['Truth_MIX'] = ((df['EQ 1000C BCC_B2'] > 0.01) & (df['EQ 1000C FCC_L12'] > 0.01)).astype(int)

    df['Truth_BCC_Bound'] = df['Truth_BCC']
    df['Truth_FCC_Bound'] = df['Truth_FCC']
    df['Truth_MIX_Bound'] = df['Truth_MIX']

    df['Truth_BCC'] = scaler.fit_transform(df['Truth_BCC'].values.reshape(-1, 1))
    df['Truth_FCC'] = scaler.fit_transform(df['Truth_FCC'].values.reshape(-1, 1))
    df['Truth_MIX'] = scaler.fit_transform(df['Truth_MIX'].values.reshape(-1, 1))



    # df_bcc = df[df['Truth_BCC'] > 0]
    # df_fcc = df[df['Truth_FCC'] > 0]
    # df_mix = df[df['Truth_MIX'] > 0]

    df_bcc = df[df['Prior_BCC'] > 0]
    df_fcc = df[df['Prior_FCC'] > 0]
    df_mix = df[df['Prior_MIX'] > 0]

    # fig, axs = plt.subplots(1, 1, figsize=(8, 6), subplot_kw={'projection': 'ternary'})
    # axs[0, 0].scatter(df_bcc['Ni'], df_bcc['Fe'], df_bcc['Cr'], c='Blue', facecolor='None', s=5)
    # axs[0, 0].scatter(df_fcc['Ni'], df_fcc['Fe'], df_fcc['Cr'], c='Red', facecolor='None', s=5)
    # axs[0, 0].scatter(df_mix['Ni'], df_mix['Fe'], df_mix['Cr'], c='Purple', facecolor='None', s=5)
    # plt.show()


    # fig = plt.figure(figsize=(6, 4))
    # ax = fig.add_subplot(1,1,1, projection="ternary")
    # pc = ax.scatter(df_fcc['Ni'], df_fcc['Fe'], df_fcc['Cr'], c = 'Green')
    # pc = ax.scatter(df_bcc['Ni'], df_bcc['Fe'], df_bcc['Cr'], c = 'Red')
    # pc = ax.scatter(df_mix['Ni'], df_mix['Fe'], df_mix['Cr'], c = 'Blue')
    #
    # plt.savefig('Prior.png')
    # plt.show()
    #


    df['Queried?'] = False
    sampled_row = df.sample(n=1)
    df.loc[sampled_row.index, 'Queried?'] = True

    #df_log = pd.DataFrame(columns=['ITR', 'Sum Error'],)
    # Active learning loop for BCC, FCC, and MIX classification

    df_log = pd.DataFrame(columns=['ITR','Error'])
    for i in range(30):

        df_queried = df[df['Queried?'] == True]
        print(f'There are {len(df_queried)} queried')

        # Prepare input (X) and output (y) for GPR fitting
        X_train = df_queried[['Ni', 'Fe', 'Cr']].values

        # Prepare targets and priors for BCC, FCC, and MIX
        y_train_bcc = df_queried['Truth_BCC'].values
        y_train_fcc = df_queried['Truth_FCC'].values
        y_train_mix = df_queried['Truth_MIX'].values

        y_train_prior_bcc = df_queried['Prior_BCC'].values
        y_train_prior_fcc = df_queried['Prior_FCC'].values
        y_train_prior_mix = df_queried['Prior_MIX'].values

        # Create ensemble GPC model and fit it
        ensemble_gpc = EnsembleGaussianProcess(kernel=RBF(length_scale_bounds=(.05,1))+WhiteKernel(noise_level_bounds=(0.001,.2)),return_std=True, normalize_y=True, n_restarts_optimizer=50,)
        ensemble_gpc.fit(X_train, y_train_bcc, y_train_fcc, y_train_mix, y_train_prior_bcc, y_train_prior_fcc, y_train_prior_mix)

        # Predict on the entire design space
        X_space = df[['Ni', 'Fe', 'Cr']].values
        y_space_prior_bcc = df['Prior_BCC'].values
        y_space_prior_fcc = df['Prior_FCC'].values
        y_space_prior_mix = df['Prior_MIX'].values

        predictions, combined_latent_std, entropy = ensemble_gpc.predict(X_space, y_space_prior_bcc, y_space_prior_fcc, y_space_prior_mix)

        # Here I am going to unscale the truth and then calculate the error.
        scaler = MinMaxScaler(feature_range=(0, 1))

        df['BCC_Prob'] = predictions[:, 0]
        df['FCC_Prob'] = predictions[:, 1]
        df['MIX_Prob'] = predictions[:, 2]

        df['BCC_Error'] = np.abs(df['Truth_BCC_Bound'] - df['BCC_Prob'])
        df['FCC_Error'] = np.abs(df['Truth_FCC_Bound'] - df['FCC_Prob'])
        df['MIX_Error'] = np.abs(df['Truth_MIX_Bound'] - df['MIX_Prob'])
        df['SUM_Error'] = df['BCC_Error'] + df['FCC_Error'] + df['MIX_Error']

        # Convert the true class columns to a single array of true labels
        df['True_Class'] = df[['Truth_BCC', 'Truth_FCC', 'Truth_MIX']].idxmax(axis=1)

        # Map true class names to numerical labels for scikit-learn metrics
        class_mapping = {'Truth_BCC': 0, 'Truth_FCC': 1, 'Truth_MIX': 2}
        df['True_Class'] = df['True_Class'].map(class_mapping)

        # Predicted probabilities as an array
        y_true = df['True_Class']
        y_pred_probs = df[['BCC_Prob', 'FCC_Prob', 'MIX_Prob']].values

        # Calculate Brier loss for each class and average
        brier_loss_bcc = brier_score_loss(df['Truth_BCC'], df['BCC_Prob'])
        brier_loss_fcc = brier_score_loss(df['Truth_FCC'], df['FCC_Prob'])
        brier_loss_mix = brier_score_loss(df['Truth_MIX'], df['MIX_Prob'])
        brier_loss_avg = (brier_loss_bcc + brier_loss_fcc + brier_loss_mix) / 3

        # Calculate log loss (cross-entropy)
        log_loss_value = log_loss(y_true, y_pred_probs)

        # Calculate accuracy, precision, recall, and F1 score using predictions
        # Convert probabilities to predicted classes (argmax approach)
        y_pred_classes = y_pred_probs.argmax(axis=1)

        precision = precision_score(y_true, y_pred_classes, average='macro')
        recall = recall_score(y_true, y_pred_classes, average='macro')
        accuracy = accuracy_score(y_true, y_pred_classes)
        f1 = f1_score(y_true, y_pred_classes, average='macro')

        # Initialize MinMaxScaler with the desired range
        scaler = MinMaxScaler(feature_range=(0.00001, 1))

        # Reshape the arrays to 2D as required by MinMaxScaler
        combined_latent_std_scaled = scaler.fit_transform(combined_latent_std.reshape(-1, 1)).flatten()
        entropy_scaled = scaler.fit_transform(entropy.reshape(-1, 1)).flatten()

        # Calculate the product
        product_of_entropy_and_uncert = entropy_scaled #combined_latent_std_scaled * entropy_scaled

        # Select the data points with the highest entropy
        next_point = acquisition_function(product_of_entropy_and_uncert)[0]
        print(f'Next point to query: {next_point}')


        df.loc[next_point, 'Queried?'] = True
        print(f'Queried point: {df.iloc[next_point]}')

        # fig = plt.figure(figsize=(5, 4))
        # ax = fig.add_subplot(1, 1, 1, projection="ternary")
        # ax.scatter(df['Ni'], df['Fe'], df['Cr'], c=predictions,s=35)
        # ax.scatter(df_queried['Ni'], df_queried['Fe'], df_queried['Cr'], c='k', facecolor='None', s=5)
        # plt.savefig(f'A_RGB_{i}.png')
        # plt.close()


        # fig = plt.figure(figsize=(5, 4))
        # ax = fig.add_subplot(1, 1, 1, projection="ternary")
        # ax.scatter(df['Ni'], df['Fe'], df['Cr'], c=df['SUM_Error'],s=35,cmap='seismic',vmin=0,vmax=2)
        # ax.scatter(df_queried['Ni'], df_queried['Fe'], df_queried['Cr'], c='k', facecolor='None', s=5)
        # plt.savefig(f'A_Errors_{i}.png')
        # plt.close()


        # # Assuming your plotting steps follow
        # # Create a figure with 2 rows and 4 columns for subplots
        # fig, axs = plt.subplots(2, 3, figsize=(20, 8), subplot_kw={'projection': 'ternary'})
        #
        # # Plot BCC predictions
        # pc = axs[0, 0].scatter(df['Ni'], df['Fe'], df['Cr'], c=predictions[:, 0], cmap='Blues', vmin=0, vmax=1)
        # axs[0, 0].scatter(df_queried['Ni'], df_queried['Fe'], df_queried['Cr'], c='k', facecolor='None', s=5)
        # cax = axs[0, 0].inset_axes([1.05, 0.1, 0.05, 0.9], transform=axs[0, 0].transAxes)
        # colorbar = fig.colorbar(pc, cax=cax)
        # colorbar.set_label("Pred BCC Prob", rotation=270, va="baseline")
        #
        # # Plot FCC predictions
        # pc = axs[0, 1].scatter(df['Ni'], df['Fe'], df['Cr'], c=predictions[:, 1], cmap='Reds', vmin=0, vmax=1)
        # axs[0, 1].scatter(df_queried['Ni'], df_queried['Fe'], df_queried['Cr'], c='k', facecolor='None', s=5)
        # cax = axs[0, 1].inset_axes([1.05, 0.1, 0.05, 0.9], transform=axs[0, 1].transAxes)
        # colorbar = fig.colorbar(pc, cax=cax)
        # colorbar.set_label("Pred FCC Prob", rotation=270, va="baseline")
        #
        # # Plot MIX predictions
        # pc = axs[0, 2].scatter(df['Ni'], df['Fe'], df['Cr'], c=predictions[:, 2], cmap='Purples', vmin=0, vmax=1)
        # axs[0, 2].scatter(df_queried['Ni'], df_queried['Fe'], df_queried['Cr'], c='k', facecolor='None', s=5)
        # cax = axs[0, 2].inset_axes([1.05, 0.1, 0.05, 0.9], transform=axs[0, 2].transAxes)
        # colorbar = fig.colorbar(pc, cax=cax)
        # colorbar.set_label("Pred MIX Prob", rotation=270, va="baseline")
        #
        # # Plot Entropy
        # pc = axs[1, 0].scatter(df['Ni'], df['Fe'], df['Cr'], c=entropy, cmap='jet',vmin=0,vmax=1.58)
        # axs[1, 0].scatter(df_queried['Ni'], df_queried['Fe'], df_queried['Cr'], c='k', facecolor='None', s=3)
        # cax = axs[1, 0].inset_axes([1.05, 0.1, 0.05, 0.9], transform=axs[1, 0].transAxes)
        # colorbar = fig.colorbar(pc, cax=cax)
        # colorbar.set_label("Shanon Entropy (bits)", rotation=270, va="baseline")
        #
        # # Plot Uncertainty (combined_latent_std)
        # pc = axs[1, 1].scatter(df['Ni'], df['Fe'], df['Cr'], c=combined_latent_std, cmap='magma_r', vmin=0,vmax=13 )
        # axs[1, 1].scatter(df_queried['Ni'], df_queried['Fe'], df_queried['Cr'], c='k', facecolor='None', s=3)
        # cax = axs[1, 1].inset_axes([1.05, 0.1, 0.05, 0.9], transform=axs[1, 1].transAxes)
        # colorbar = fig.colorbar(pc, cax=cax)
        # colorbar.set_label("Latent Uncertainty ()", rotation=270, va="baseline")
        #
        # # Plot Product of Entropy and Uncertainty
        # pc = axs[1, 2].scatter(df['Ni'], df['Fe'], df['Cr'], c=df['SUM_Error'], cmap='seismic') #vmin=0,vmax=14
        # axs[1, 2].scatter(df_queried['Ni'], df_queried['Fe'], df_queried['Cr'], c='k', facecolor='None', s=3)
        # cax = axs[1, 2].inset_axes([1.05, 0.1, 0.05, 0.9], transform=axs[1, 2].transAxes)
        # colorbar = fig.colorbar(pc, cax=cax)
        # colorbar.set_label("Entropy x Uncertainty (scaled bits)", rotation=270, va="baseline")
        #
        # # Adjust layout to prevent overlap and make space for labels
        # plt.tight_layout()
        #
        # # # Save the updated figure with all subplots
        # plt.savefig(f'ITR_AQF_combined_with_entropy_uncert_{i}.png')
        #
        # # Close the figure to free memory
        # plt.close(fig)

        df_error_metrics.at[i+1,'ITR'] = i
        df_error_metrics.at[i+1,'Brier_Loss_{}'.format(j)] = brier_loss_avg
        df_error_metrics.at[i+1,'Log_Loss_{}'.format(j)] = log_loss_value
        df_error_metrics.at[i+1,'Accuracy_{}'.format(j)] = accuracy
        df_error_metrics.at[i+1,'Precision_{}'.format(j)] = precision
        df_error_metrics.at[i+1,'Recall_{}'.format(j)] = recall
        df_error_metrics.at[i+1,'F1_{}'.format(j)] = f1

        print(i)
        df_log.at[i+1,'ITR'] = i
        df_log.at[i+1,'Error'] = df['SUM_Error'].sum(axis=0)
        df_log.at[i+1,'Brier_Loss'] = brier_loss_avg
        df_log.at[i+1,'Log_Loss'] = log_loss_value
        df_log.at[i+1,'Accuracy'] = accuracy
        df_log.at[i+1,'Precision'] = precision
        df_log.at[i+1,'Recall'] = recall
        df_log.at[i+1,'F1'] = f1
        df_log.to_csv('/data/log_No_prior.csv')
        #df.to_csv('output_No_Prior.csv')
    df_error_metrics.to_csv('/data/Error_Metric_wo_Prior.csv')