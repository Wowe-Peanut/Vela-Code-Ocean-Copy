import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from collections import defaultdict
from sklearn.gaussian_process import GaussianProcessRegressor
from math import log
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, log_loss


# Inverse sigmoid function
def logit(sigma):
    return np.log(sigma / (1 - sigma))

# Multi-class brier loss calculator
def mc_brier_loss(ground_truths, predicted_probs):
    total = 0

    for probs, gt in zip(predicted_probs, ground_truths):
        total += np.mean([(probs[i] - int(i==gt))**2 for i in range(len(probs))])

    return total/len(predicted_probs)

# Converts list of classes into binary vector with 1 only if class = target
def one_hot(vector, target):
    return [int(v == target) for v in vector]

# Util class for storing the results of test splits and calculating/plotting error metrics
class ModelEvaluator:

    def __init__(self, classes):
        self.splits_ = defaultdict(list)

        self.classes_ = classes
        self.ntoc_ = dict([(name,cls) for cls,name in enumerate(classes)])

    def add_split(self, model_name, ground_truths, preds=None, probs=None):
        self.splits_[model_name].append((ground_truths, preds, probs))

    # returns in a "model_name: [ac,pr,rc,bl,f1,ll]" dictionary
    def calc_errors(self, target, avg_method, zero_div_method):
        error_dict = dict()

        for model_name, splits in self.splits_.items():
            errors = [[],[],[],[],[],[]]

            for gts, preds, probs in splits:
                # If only class probabilities are passed, take max prob class to be the prediction for each datapoint
                if preds is None:
                    preds = np.argmax(probs, axis=-1)

                # If only class predictions are passed, assign class probability to 100 for the prediction, 0 otherwise
                if probs is None:
                    probs = np.full((len(gts), len(self.classes_)), 0)
                    for i, pred in enumerate(preds):
                        probs[i][pred] = 1

                # For targeted error metrics, one-hot the targeted class, and sum probabilities accordinging
                if target != 'ALL':
                    preds = one_hot(preds, self.ntoc_[target])
                    gts = one_hot(gts, self.ntoc_[target])

                    target_id = self.ntoc_[target]
                    probs = [[prob[target_id], 1-prob[target_id]] for prob in probs]


                # NOTE: other methods are expecting the order [ac,pr,rc,bl,f1,ll]
                errors[0].append(accuracy_score(gts, preds))
                errors[1].append(precision_score(gts, preds, average=avg_method, zero_division=zero_div_method))
                errors[2].append(recall_score(gts, preds, average=avg_method, zero_division=zero_div_method))
                errors[3].append(mc_brier_loss(gts, probs))
                errors[4].append(f1_score(gts, preds, average=avg_method, zero_division=zero_div_method))
                errors[5].append(log_loss(gts, probs))

            error_dict[model_name] = errors
        return error_dict

    def print_summary(self, target='ALL', avg_method="weighted", zero_div_method=0):
        model_errors = self.calc_errors(target, avg_method, zero_div_method)

        print("\n\n------------------------------------------------")
        for model_name, (ac,pr,rc,bl,f1,ll) in model_errors.items():
            print(f"Average '{model_name}' Error ({target} Alloys):")
            print(f"    Accuracy:\t\t{np.mean(ac):.4f}\t| std {np.std(ac):.4f}")
            print(f"    Precision:\t\t{np.mean(pr):.4f}\t| std {np.std(pr):.4f}")
            print(f"    Recall:\t\t\t{np.mean(rc):.4f}\t| std {np.std(rc):.4f}")
            print(f"    F1-Score:\t\t{np.mean(f1):.4f}\t| std {np.std(f1):.4f}")
            print(f"    MC Brier Loss:\t{np.mean(bl):.4f}\t| std {np.std(bl):.4f}")
            print(f"    Log Loss:\t\t{np.mean(ll):.4f}\t| std {np.std(ll):.4f}")

            print()

    def whisker_plots(self, title, model_order=None, avg_method="weighted", zero_div_method=0,
                            show=True, save_as=None, target='ALL', dpi=500):

        # If order of models in each subplot is not specified, use order in split dictionary
        if model_order is None:
            model_order = self.splits_.keys()

        # Calculate errors for each split
        model_errors = self.calc_errors(target, avg_method, zero_div_method)

        # Setup dataframe with column names labeled with each combination of model name and error type
        error_df = pd.DataFrame({})
        for i, (model_name, (ac,pr,rc,bl,f1,ll)) in enumerate(model_errors.items()):
            error_df[f"{model_name} AC"] = ac
            error_df[f"{model_name} PR"] = pr
            error_df[f"{model_name} RC"] = rc
            error_df[f"{model_name} F1"] = f1
            error_df[f"{model_name} LL"] = ll

            if len(bl) != 0:
                error_df[f"{model_name} BL"] = bl


        # Plot all the whisker subplots
        fig, axes = plt.subplots(2, 3)
        axes[0, 0].set_title(f"Accuracy")
        sb.boxplot(error_df[[f"{model} AC" for model in model_order]], ax=axes[0, 0], showfliers=False)
        axes[0, 0].set_xticks([0,1,2])
        axes[0, 0].set_xticklabels(['Uninf.', 'TC', 'Inf.'])

        axes[0, 1].set_title(f"Precision")
        sb.boxplot(error_df[[f"{model} PR" for model in model_order]], ax=axes[0, 1], showfliers=False)
        axes[0, 1].set_xticks([0, 1, 2])
        axes[0, 1].set_xticklabels(['Uninf.', 'TC', 'Inf.'])

        axes[0, 2].set_title(f"MC Brier Loss")
        sb.boxplot(error_df[[f"{model} BL" for model in model_order]], ax=axes[0, 2], showfliers=False)
        axes[0, 2].set_xticks([0, 1, 2])
        axes[0, 2].set_xticklabels(['Uninf.', 'TC', 'Inf.'])

        axes[1, 0].set_title(f"Recall")
        sb.boxplot(error_df[[f"{model} RC" for model in model_order]], ax=axes[1, 0], showfliers=False)
        axes[1, 0].set_xticks([0, 1, 2])
        axes[1, 0].set_xticklabels(['Uninf.', 'TC', 'Inf.'])

        axes[1, 1].set_title(f"F1-Score")
        sb.boxplot(error_df[[f"{model} F1" for model in model_order]], ax=axes[1, 1], showfliers=False)
        axes[1, 1].set_xticks([0, 1, 2])
        axes[1, 1].set_xticklabels(['Uninf.', 'TC', 'Inf.'])

        axes[1, 2].set_title(f"Log Loss")
        sb.boxplot(error_df[[f"{model} LL" for model in model_order]], ax=axes[1, 2], showfliers=False)
        axes[1, 2].set_xticks([0, 1, 2])
        axes[1, 2].set_xticklabels(['Uninf.', 'TC', 'Inf.'])

        fig.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        if save_as is not None:
            plt.savefig(save_as, dpi=dpi)
        if show:
            plt.show()

# Custom Gaussian Process Classifier made of one vs. rest regressors to incorporate priors
class GPClassifier:

    """
    Constructor
    Allow for all the same parameters as a normal sklearn GPR/GPC but allows you to specify
    how many classes you are classifying and assign priors.
    """
    def __init__(self, numClasses, kernel=None, priors = None, *, alpha=1e-10,
                 optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0,
                 normalize_y=False, classPositive=1):

        self.regs_ = [GaussianProcessRegressor( kernel=kernel,
                                                alpha = alpha,
                                                optimizer=optimizer,
                                                n_restarts_optimizer=n_restarts_optimizer,
                                                normalize_y=normalize_y) for _ in range(numClasses)]
        self.kernel_ = kernel
        self.classPositive_ = classPositive
        self.numClasses_ = numClasses
        self.priors_ = priors if priors else ([lambda x: 0]*numClasses)


    """
    Sigmoid function used to bound predictions between 0 and 1 to generate
    a probability value rather than just a scaler value.
    """
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    """
    Softmax function can be used to normalize multiple joint probabilities.
    Used in .predict_proba()
    """
    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


    """
    Internal helper function: applies the saved priors to each vector passed in to generate a prior
    probability for each.
    Used in .fit() and .predict_proba()
    """
    def prior_predict_(self, x):
        n = len(x)
        p = np.zeros((n, self.numClasses_))

        for classNum in range(self.numClasses_):
            for i in range(n):
                p[i][classNum] = self.priors_[classNum](x[i])

        return p


    """
    Fits each internal GPR to identify a single unique class each. It incorporates the
    saved priors by training each GPR on the difference between the observed values and
    the prior beliefs. Allows you to pass a discrete prior probability matrix instead
    of providing a mathmatical function.
    
    X:              Feature vectors
    y:              1D list of the class of each feature vector, must be in range [0, NumberOfClasses)
    prior:          Prior probabilities of each feature vector in X. If not specified, the prior functions
                    specified during construction or the default priors are used.
    returns:        None
    """
    def fit(self, X, y, prior=None):
        n = len(X)
        if n == 0:
            raise RuntimeError("Model cannot fit zero on empty arrays")
        if len(X) != len(y):
            raise RuntimeError("Feature vectors array and target array must be the SAME SIZE")


        # If the prior probabilities for this input are not provided, use the
        # prior created during construction
        priorPredictions = np.array(prior if prior is not None else self.prior_predict_(X))

        # For each "one vs rest" GPR register, we train on the
        # difference between the ground truth and the prior assumption
        for classNum, reg in enumerate(self.regs_):
            truth = np.array([(self.classPositive_ if y[i] == classNum else -self.classPositive_) for i in range(n)])
            beta = truth - priorPredictions[:, classNum]
            beta = beta.reshape(-1, 1)
            reg.fit(X, beta)


    """
    Returns the normalized probability of each given vector in x belonging to each class the model is fitted for.
    The GPRs predict how far off each probability is from the prior belief, before normalizing the probabilities
    with either marginal probabilities or using softmax.
    
    x:              feature vectors who's class probabilities will be calculated
    prior:          Prior probabilities of each feature vector in x. If not specified, the prior functions
                    specified during construction or the default priors are used.
    returns:        list of vectors containing the probability of each vector being of each of the particular classes
    """
    def predict_proba(self, x, prior=None, useSoftmax=True):
        n = len(x)
        if n == 0:
            raise RuntimeError("Model cannot predict on empty arrays")

        # Calculate initial prediction probabilities
        predictedOffset = np.array([reg.predict(x) for reg in self.regs_])
        priorPredictions = np.array(prior if prior is not None else self.prior_predict_(x))

        # Add the prior predictions + the offset predicted by each GPR and pass through sigmoid
        preds = np.array([self.sigmoid(predictedOffset[i] + priorPredictions[:, i]) for i in range(self.numClasses_)])


        # Normalize the probabilities and return
        probs = np.zeros((n, self.numClasses_))
        for i in range(n):
            p = np.array([preds[j][i] for j in range(self.numClasses_)])

            # Uses either softmax or marginal probabilities to normalize the predictions
            if useSoftmax:
                probs[i] = self.softmax(p)
            else:
                probs[i] = p

        return probs


    """
    Predicts the class from [0,numClasses), of the given feature vectors. Each is labeled by the 
    class it is MOST LIKELY to belong to of all the classes after normalization.
    """
    def predict(self, x, prior=None, useSoftmax=True):
        return np.argmax(self.predict_proba(x, prior, useSoftmax), axis=-1)


    """
    Finds the index of the feature vector in x with the highest shannon entropy: the point with the
    most uncertainty. Useful for active learning
    """
    def find_entropic_max(self, x, prior=None):
        if len(x) == 0:
            raise RuntimeError("Cannot find the max entropy input in an empty array...")

        probs = self.predict_proba(x, prior)

        entropyVals = [-sum([p*log(p) for p in probVec]) for probVec in probs]
        highest = entropyVals[0]
        loc = 0

        for i in range(len(entropyVals)):
            if entropyVals[i] > highest:
                highest = entropyVals[i]
                loc = i

        return loc
