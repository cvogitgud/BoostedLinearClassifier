# BoostedLinearClassifier

### boostit.py
**Important functions** - *fit(), predict()*
  
Implements a boosting algorithm as a class with a cluster means classifier as the base learner. The cluster means classifier is a parametric supervised learning model (training points already have known labels, and are grouped by their labels). The model calculates the weighted mean of each class of a set of training data (train.npy), and then assigns to a test point the class of the closest class centroid as the prediction. In practice, the very same training set is used since it is non-linearly separable data. The boosting algorithm iteratively creates a model ensemble, then takes the weighted average of those models to use on the final test set. Model ensemble techniques like boosting and bagging allow us to decrease variance without increasing bias.
  
While local_evaluation.py is included as a separate file, I included local_evaluation.py as the main() function to measure performance with every run. fit() also outputs performance stats for each iteration/model produced. These are optional, but helpful in validation set testing and debugging.

### dataset1
Folder of the training set and the test set. Both are .npy files.

### local_evaluation.py
Calculates and outputs various performance stats for the boosted cluster means classifier. 
