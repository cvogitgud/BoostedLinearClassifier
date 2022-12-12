import numpy as np

class BoostingClassifier:

    def __init__(self):
        self.M_x = []       # weighted average of ensemble of centroids
        self.T = 5          # model count; hyper-parameter

    def fit(self, X, y):
        """ Fit the boosting model.

        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)
            The input samples with dtype=np.float32.

        y : { numpy.ndarray } of shape (n_samples,)
            Target values. By default, the labels will be in {-1, +1}.

        Returns
        -------
        self : object
        """
        w = [1/len(X)] * len(X)                 # initialize weights
        ensemble = []                           # centroids from each model
        confidence_factors = []
        for t in range(self.T):
            print('Iteration %i:' % (t + 1))
            # run base learner algorithm
            means = centroids(X, y, w)
            predictions_t = basic_linear_classifier(means, X)

            # calculate weighted error
            mclf = []                           # instances misclassified this round
            w_m = []                            # weights of misclassified instances
            for i in range(len(predictions_t)):
                if predictions_t[i] != y[i]:
                    mclf.append(i)
                    w_m.append(w[i])
            error_rate = np.sum(w_m)
            print('Error rate = %s' % round(error_rate, 4))

            if error_rate >= 0.5:
                break

            # weight adjustments
            confidence = 0.5 * np.log((1 - error_rate) / error_rate)
            print('Alpha = %s' % round(confidence, 4))

            confidence_factors.append(confidence)
            mclf_weight = 1 / (2 * error_rate)
            cclf_weight = 1 / (2 * (1 - error_rate))
            for i in range(len(X)):
                if i in mclf:
                    w[i] = w[i] * mclf_weight
                else:
                    w[i] = w[i] * cclf_weight
            print('Factor to increase weights = %s' % round(mclf_weight, 4))
            print('Factor to decrease weights = %s' % round(cclf_weight, 4))

            ensemble.append(confidence * np.array(means))

        # compute weighted average of our model ensemble, save that as our new weighted average model
        self.M_x = np.sum(ensemble, axis=0) / sum(confidence_factors)

        return self

    def predict(self, X):
        """ Predict binary class for X.

        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)

        Returns
        -------
        y_pred : { numpy.ndarray } of shape (n_samples)
                 In this sample submission file, we generate all ones predictions.
        """
        return basic_linear_classifier(self.M_x, X)


#--------------------------------------------------------------------------------------------------
# Helper functions for fit()

# Function: Splits training data according to class
#           Helper for basic_linear_classifier()
# Output: 2 arrays of set of points of each class
# Note: train_data and train_labels are numpy arrays, read into and from fit()
# IMPORTANT: Altered to pair point with its weight to calculate weighted centroids in centroids()
def class_arr(train_data, train_labels, weights):
    # class0 = -1 class
    class0 = []
    class1 = []
    for i in range(len(train_data)):
        if train_labels[i] == -1:
            class0.append([train_data[i], weights[i]])
        else:
            class1.append([train_data[i], weights[i]])
    return class0, class1

# Input: 2 numpy arrays (X's points have weights applied before call)
#        X = data points, Y = training labels
# Output: array of centroids for each class
# Function: calculates the WEIGHTED centroids for 2 classes
def centroids(X, Y, weights):
    classes = class_arr(X, Y, weights)            # point-weight pairs by class
    centroid_arr = []
    for i in range(2):
        sum_points = 0                            # sum of weighted points in class
        sum_weights = 0                           # sum of weights of points in class
        for p, w in classes[i]:
            sum_points += p * w
            sum_weights += w

        # compute weighted class mean/centroid
        class_centroid = sum_points / sum_weights
        centroid_arr.append(class_centroid)
    return centroid_arr

# Input: centroids, test data set
# Output: a numpy array of the predicted class for every data point
# Function: calculates distance of point from each class centroid;
#           minimum distance is predicted class
#           - base learner for fit()
def basic_linear_classifier(centroid_arr, test_data):
    predictions = np.array([])

    # centroid0, dist0 are for label -1
    centroid0 = centroid_arr[0]
    centroid1 = centroid_arr[1]
    for point in test_data:
        dist0 = np.linalg.norm(centroid0 - point)
        dist1 = np.linalg.norm(centroid1 - point)

        # min. distance = predicted class
        if np.amin([dist0, dist1]) == dist0:
            predictions = np.append(predictions, -1)
        else:
            predictions = np.append(predictions, 1)
    return predictions

if __name__ == "__main__":
    #!/usr/bin/env python3
    import sys
    import os
    import numpy as np
    import time

    # evaluation on your local machine only
    dataset_dir = 'dataset1'
    train_set = os.path.join(dataset_dir, 'train.npy')
    test_set = os.path.join(dataset_dir, 'test.npy')

    def evaluation_score(y_pred, y_test):
        y_pred = np.squeeze(y_pred)
        assert y_pred.shape == y_test.shape, "Error: the shape of your prediction doesn't match the shape of ground truth label."

        TP = 0	# truth positive
        FN = 0	# false negetive
        TN = 0	# true negetive
        FP = 0 	# false positive

        for i in range(len(y_pred)):
            pred_label = y_pred[i]
            gt_label = y_test[i]

            if int(pred_label) == -1:
                if pred_label == gt_label:
                    TN += 1
                else:
                    FN += 1
            else:
                if pred_label == gt_label:
                    TP += 1
                else:
                    FP += 1

        accuracy = round((TP + TN) / (TP + FN + FP + TN),4)
        precision = round(TP / (TP + FP) if ((TP + FP) > 0) else 0,4)
        recall = round(TP / (TP + FN) if ((TP + FN)) > 0 else 0,4)
        f1 = round(2 * precision * recall / (precision + recall) if ((precision + recall) > 0) else 0,4)
        final_score = round(50 * accuracy + 50 * f1,4)

        print("\nTesting:")
        print("TP: {}\nFP: {}\nTN: {}\nFN: {}\nError rate: {}".format(TP, FP, TN, FN, (FP + FN) / (TP + FP + TN + FN)))
        return accuracy, precision, recall, f1, final_score

    # load dataset
    with open(train_set, 'rb') as f:
        X_train = np.load(f)
        y_train = np.load(f)

    with open(test_set, 'rb') as f:
        X_test = np.load(f)
        y_test = np.load(f)

    clf = BoostingClassifier().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc, precision, recall, f1, final_score = evaluation_score(y_pred, y_test)

    print("Accuracy: {}, F-measure: {}, Precision: {}, Recall: {}, Final_Score: {}".format(acc, f1, precision, recall, final_score))