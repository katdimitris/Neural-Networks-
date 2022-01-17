from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


def grid_search(X_train, y_train, X_test, y_test):
    tuned_parameters = [
        {"kernel": ["rbf"],
         "gamma": [0.2, 0.1, 0.05, 0.01, 0.005],
         "C": [5, 10, 15, 20]}]

    # split training set into training and validation subsets
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    #Define the scoring measures for evaluating svm's perfomance
    scores = ["f1"]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, scoring="%s_macro" % score, cv=5)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set using 5-split cross validation :")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()












