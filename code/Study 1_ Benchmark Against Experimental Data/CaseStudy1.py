import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sys import exit
from collections import Counter
import time

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, Matern, RationalQuadratic
from gpc_utils import *

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning





##################################################################################
# PREPROCESSING
##################################################################################
start_time = time.perf_counter()

# Load in the featurized and prior predicted dataframe
df = pd.read_csv('/data/CS1_ProcessedMachakaDataset.csv')

# Define that classes being classified
classes = ["FCC", "FCC + Im", "BCC", "BCC + Im"]
ntoc = dict([(name, cls) for cls, name in enumerate(classes)])  # Class name to class number

# Labels each datapoint based on thermo-calc's predicted FCC and BCC percentages
df.loc[df['HOMO FCC SUM'] >= 0.50, 'TC Predicted Class Name']    = 'FCC + Im'
df.loc[df['HOMO FCC SUM'] >= 0.99, 'TC Predicted Class Name']    = 'FCC'
df.loc[df['HOMO BCC SUM'] >= 0.50, 'TC Predicted Class Name']    = 'BCC + Im'
df.loc[df['HOMO BCC SUM'] >= 0.99, 'TC Predicted Class Name']    = 'BCC'

# Drop rows TC isn't confident enough to put into any of the classes.
df = df[~df['TC Predicted Class Name'].isna()]

# Then convert the class labeled by TC into numerical values that we can plug into the GPC
df['TC Predicted Class'] = df['TC Predicted Class Name'].map(ntoc).astype(int)
df['GT Class'] = df['Microstructure'].map(ntoc)






##################################################################################
# MODEL & EXECUTION PARAMETERS
##################################################################################

# Ignore scikit-learn convergence warnings
simplefilter("ignore", category=ConvergenceWarning)

splits      = 500
test_size   = 0.8
stratified  = True

prior_weights = logit(np.array([
    [0.5, 0.4, 0.05, 0.05],
    [0.4, 0.5, 0.05, 0.05],
    [0.05, 0.05, 0.5, 0.4],
    [0.05, 0.05, 0.4, 0.5]
]))

feats = ['Yang delta', 'Yang omega', 'APE mean', 'Radii local mismatch', 'Radii gamma',
         'Configuration entropy', 'Atomic weight mean', 'Total weight', 'Lambda entropy', 'Electronegativity delta']

gpc = GPClassifier(numClasses           = len(classes),
                   kernel               = RBF(length_scale=[1] * len(feats)) + WhiteKernel(),
                   classPositive        = 5,
                   n_restarts_optimizer = 10)




##################################################################################
# TRAIN/TEST
##################################################################################

# Feature vectors, Thermo-calc's prior probability predictions, and ground truths respectively
X = df[feats].values  
prior_probabilities = prior_weights[df['TC Predicted Class'].values]  
y = df['GT Class'].values  
print("Full Count:", sorted(Counter(y).items(), key=lambda x: x[0]))


# If set, train/test splits will roughly preserve class distribution
stratify = y[:] if stratified else None


# Custom model error calculation tool
eval = ModelEvaluator(classes=classes)


# Performs multiple train/test splits and calculates the error for each
for split in range(splits):

    # Split data into train & test subsets
    (X_train, X_test,
     prior_probs_train, prior_probs_test,
     y_train, y_test) = train_test_split(X, prior_probabilities, y, test_size=test_size, stratify=stratify)

    ##### UNINFORMED -----------------------------------------------------------
    gpc.fit(X_train, y_train, prior=None)
    model_probs = gpc.predict_proba(X_test, prior=None)
    eval.add_split('Uninf.', y_test, probs=model_probs)

    ##### INFORMED -----------------------------------------------------------
    gpc.fit(X_train, y_train, prior=prior_probs_train)
    model_probs = gpc.predict_proba(X_test, prior=prior_probs_test)
    eval.add_split('Inf.', y_test, probs=model_probs)

    ##### TC Prediction ------------------------------------------------------
    model_preds = np.argmax(prior_probs_test, axis=-1)
    eval.add_split('TC', y_test, preds=model_preds)

    if (split+1)%10 == 0:
        print(f"{split+1}/{splits} splits complete")





##################################################################################
# ERROR CALCULATIONS
##################################################################################
folder = '/results/'

eval.print_summary(target='ALL')
for target in ['ALL', "FCC", "FCC + Im", "BCC", "BCC + Im"]:
    eval.whisker_plots(title=f"{target} Error Metrics",
                       target=target,
                       show=True,
                       save_as=f"{folder}{target}",
                       model_order=["Uninf.", "TC", "Inf."],
                       avg_method='weighted')

print(Counter(df['Microstructure'].values))
print(f"Total Runtime: --- {time.perf_counter() - start_time:.2f} seconds ---")

