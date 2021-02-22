from sklearn import datasets, model_selection, tree
from sklearn.naive_bayes import GaussianNB
import numpy as np


def classify(classifier, x_train, x_test, cls_train, cls_test_actual):
    classifier.fit(x_train, cls_train)
    cls_test_predicted = classifier.predict(x_test)
    return np.c_[x_test, cls_test_predicted, cls_test_actual]


if __name__ == '__main__':
    iris = datasets.load_iris()
    data_set = model_selection.train_test_split(iris.data, iris.target, test_size=0.3)
    header = "sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm,class_predicted,class_actual"
    output_format = "%1.1f,%1.1f,%1.1f,%1.1f,%d,%d"
    results = {'naive_bayes.csv': classify(tree.DecisionTreeClassifier(), *data_set),
               'decision_tree.csv': classify(GaussianNB(), *data_set)}
    for file_name, result in results.items():
        np.savetxt(file_name, result, output_format, header=header, comments='')
