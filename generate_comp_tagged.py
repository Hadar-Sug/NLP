import pickle
import numpy as np
import os
import time
import pandas as pd
from collections import Counter
import sys
sys.path.append("code")
from code.preprocessing import preprocess_train
from code.optimization import get_optimal_vector
from code.inference import tag_all_test


NUM_FEATURES = 15

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(CURR_DIR,'code')
SUB_DIR = os.path.join(CODE_DIR,'Submission')
PARENT_DIR = os.path.abspath(os.path.join(CURR_DIR, os.pardir))
DATA_DIR = os.path.join(PARENT_DIR,'data')

def calc_score_confusion_mat(true_file, pred_file, verbose=True):
    """
    Compare the true and predicted tags in two files, optionally print misclassified tags and confusion matrix,
    and calculate accuracy.

    Args:
    - true_file (str): Path to the file containing true tags.
    - pred_file (str): Path to the file containing predicted tags.
    - verbose (bool): Whether to print misclassified tags and confusion matrix (default=True).

    Returns:
    - accuracy (float): The accuracy of the model.
    """

    # Read true and predicted tags from files
    with open(true_file, 'r') as f:
        true_data = [x.strip() for x in f.readlines() if x != '']
    with open(pred_file, 'r') as f:
        pred_data = [x.strip() for x in f.readlines() if x != '']

    # Check if the number of sentences in the true and predicted data match
    if len(pred_data) != len(true_data):
        if len(pred_data) > len(true_data):
            pred_data = pred_data[:len(true_data)]
        else:
            raise KeyError("Number of sentences in true and predicted files don't match")

    # Initialize counters for correct predictions and misclassified tags
    num_correct, num_total = 0, 0
    misclassified_tags = Counter()

    true_labels = []  # List to store true labels

    # Iterate through each sentence in the data
    for idx, sen in enumerate(true_data):
        pred_sen = pred_data[idx]

        # Process true and predicted tags
        true_words = [x.split('_')[0] for x in sen.split()]
        true_tags = [x.split('_')[1] for x in sen.split()]
        true_labels.extend(true_tags)  # Extend true_labels with true_tags

        pred_words = [x.split('_')[0] for x in pred_sen.split()]
        try:
            pred_tags = [x.split('_')[1] for x in pred_sen.split()]
        except IndexError:
            pred_tags = []
            for x in pred_sen.split():
                if '_' in x:
                    pred_tags.append(x.split('_'))
                else:
                    pred_tags.append(None)
        if pred_words[-1] == '~':
            pred_words = pred_words[:-1]
            pred_tags = pred_tags[:-1]

        # Skip if sentence lengths or words are different
        if pred_words != true_words:
            continue

        # Compare true and predicted tags
        for i, (tt, tw) in enumerate(zip(true_tags, true_words)):
            num_total += 1
            if len(pred_words) > i:
                pw = pred_words[i]
                pt = pred_tags[i]
            else:
                continue
            if pw != tw:
                continue
            if tt == pt:
                num_correct += 1
            else:
                # Increment the count of misclassified tags
                misclassified_tags[(tt, pt)] += 1

    # Calculate accuracy
    accuracy = num_correct / num_total

    if verbose:
        # Get the top ten misclassified tags
        top_ten_misclassified = misclassified_tags.most_common(10)

        # Print the top ten misclassified tags
        print("Top Ten Misclassified Tags:")
        for tags, count in top_ten_misclassified:
            true_tag, pred_tag = tags
            print(f"True Tag: {true_tag}, Predicted Tag: {pred_tag}, Count: {count}")

        # Create a DataFrame for the confusion matrix
        confusion_matrix = pd.DataFrame(0, index=sorted(set(true_labels)), columns=sorted(set(true_labels)))

        # Update the confusion matrix with counts
        for (true_tag, pred_tag), count in misclassified_tags.items():
            confusion_matrix.loc[true_tag, pred_tag] = count

        # Filter confusion matrix to include only relevant rows and columns
        relevant_tags = set([tag for tags, _ in top_ten_misclassified for tag in tags])
        relevant_labels = sorted(relevant_tags)

        # Filter relevant_labels to include only labels present in the DataFrame's index
        relevant_labels = [label for label in relevant_labels if label in confusion_matrix.index]

        # Now subset the DataFrame
        relevant_confusion_matrix = confusion_matrix.loc[relevant_labels, relevant_labels]

        # Print the confusion matrix
        print("\nConfusion Matrix:")
        print(relevant_confusion_matrix)

    # Return accuracy
    return accuracy


def train(train_path, thresholds_arr, weights_path, lam):
    """
    Trains a model using the specified training data and parameters.

    Args:
    - train_path (str): Path to the training data.
    - thresholds_arr (list): List of thresholds for feature selection.
    - weights_path (str): Path to save the trained model weights.
    - lam (float): Regularization parameter.

    Returns:
    - None
    """
    # Preprocess the training data
    statistics, feature2id = preprocess_train(train_path, thresholds_arr)
    # Train the model and obtain optimal parameters
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)


def test(test_path, weights_path, predictions_path, beam_width):
    """
    Tests a trained model using the specified test data.

    Args:
    - test_path (str): Path to the test data.
    - weights_path (str): Path to the trained model weights.
    - predictions_path (str): Path to save the predicted outputs.
    - beam_width (int): Width of the beam for beam search.

    Returns:
    - None
    """
    # Load the trained model weights
    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]
    # Make predictions on the test data
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path, beam_width)


class Submit:
    """
    Class to submit trained models and obtain predictions.
    """

    def __init__(self, _lambda, thresholds, beam):
        """
        Initialize Submit object with lambda, thresholds, and beam parameters.

        Args:
        - _lambda (float): Regularization parameter.
        - thresholds (list): List of thresholds for feature selection.
        - beam (int): Width of the beam for beam search.

        Returns:
        - None
        """
        self._lambda = _lambda
        self.beam = beam
        self.thresholds = thresholds

    def fit_predict_test1(self,fit):
        """
        Train a model and make predictions on test1 data.

        Returns:
        - None
        """
        if not os.path.exists('Submission'):
            os.makedirs('Submission')
        model1_weights_path = os.path.join(SUB_DIR,'model1.pkl')
        model1_predictions_path =os.path.join(SUB_DIR,'model1_test1_pred.wtag')
        # Train the model on test1 data
        if fit:
            start_time = time.time()
            train_path = os.path.join(DATA_DIR,'train1.wtag')
            train(train_path=train_path, thresholds_arr=self.thresholds,
                  weights_path=model1_weights_path, lam=self._lambda)
            end_time = time.time()
            print(f"Training time: {end_time - start_time} seconds")
        test1_path = os.path.join(DATA_DIR,'test1.wtag')
        # Test the model on test1 data and print score
        test(test_path=test1_path, weights_path=model1_weights_path, predictions_path=model1_predictions_path,
             beam_width=self.beam)
        score = calc_score_confusion_mat(test1_path, model1_predictions_path,verbose=True)
        print(f'model1 test score: {score:.4f}')

    def fit_predict_comp1(self, fit=False):
        """
        Train a model and make predictions on comp1 data.

        Args:
        - fit (bool): Whether to train the model.

        Returns:
        - None
        """
        if not os.path.exists('Submission'):
            os.makedirs('Submission')
        model2_weights_path = os.path.join(SUB_DIR,'model2.pkl')
        model2_predictions_path = os.path.join(SUB_DIR,'comp_m1_318155843_206567067.wtag')
        if fit:
            # Train the model on comp1 data
            start_time = time.time()
            train_data =  os.path.join(DATA_DIR,'train_comp1_comp2.wtag')
            train(train_path=train_data, thresholds_arr=self.thresholds,
                  weights_path=model2_weights_path, lam=self._lambda)
            end_time = time.time()
            print(f"Training time: {end_time - start_time} seconds")
        comp1_path = os.path.join(DATA_DIR,'comp1.words')
        # Test the model on comp1 data
        test(test_path=comp1_path, weights_path=model2_weights_path, predictions_path=model2_predictions_path,
             beam_width=self.beam)

    def fit_predict_comp2(self, shrink_model=False, predict_train=False):
        """
        Train a model and make predictions on comp2 data.

        Args:
        - shrink_model (bool): Whether to shrink the model.
        - predict_train (bool): Whether to make predictions on train2 data.

        Returns:
        - None
        """
        if not os.path.exists('Submission'):
            os.makedirs('Submission')
        model2_weights_path = os.path.join(SUB_DIR,'model2.pkl')
        model3_predictions_path = os.path.join(SUB_DIR,'comp_m2_318155843_206567067.wtag')
        chosen_model_path = model2_weights_path
        train2_path = os.path.join(DATA_DIR,'train2.wtag')
        if shrink_model:
            model3_weights_path = os.path.join(SUB_DIR,'model3_shrunken.pkl')
            # Train a shrunken model on train2 data
            train(train_path=train2_path, thresholds_arr=self.thresholds,
                  weights_path=model3_weights_path, lam=self._lambda)
            chosen_model_path = model3_weights_path
        if predict_train:
            train2_pred_path = os.path.join(SUB_DIR,'model3_train2_pred.wtag')
            # Test the model on train2 data and print score
            test(test_path=train2_path, weights_path=chosen_model_path, predictions_path=train2_pred_path,
                 beam_width=self.beam)
            score = calc_score_confusion_mat(train2_path, train2_pred_path,verbose=False)
            print(f'small model train score: {score:.4f}')
        comp2_path = os.path.join(DATA_DIR,'comp2.words')
        # Test the model on comp2 data
        test(test_path=comp2_path, weights_path=chosen_model_path, predictions_path=model3_predictions_path,
             beam_width=self.beam)


def main():
    submit = Submit(0.9, np.ones(NUM_FEATURES), 2)
    # submit.fit_predict_test1(fit=False)
    submit.fit_predict_comp1(fit=False)  # submit with fit= False as wit was requested to generate the comp files
    # using the pretrained weights
    submit.fit_predict_comp2()


if __name__ == '__main__':
    main()
