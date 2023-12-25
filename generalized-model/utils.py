from sklearn.model_selection import StratifiedShuffleSplit

def balanced_data_split(X, y, max_iter: int = 5):
    # Create an instance of StratifiedShuffleSplit
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    stratified_split.get_n_splits(X, y)
    
    i = 0
    label_ratio_train = 0
    label_ratio_test = 1
    while abs(label_ratio_train-label_ratio_test)/label_ratio_test > 0.05 and i < max_iter:

        # Use the split method to generate indices for training and test sets
        for train_index, test_index in stratified_split.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        label_ratio_train = len(y_train[y_train['Survived']==0]) / len(y_train[y_train['Survived']==1])
        label_ratio_test = len(y_test[y_test['Survived']==0]) / len(y_test[y_test['Survived']==1])

        i+=1
    
    if i == max_iter:
        raise ValueError(f"Label ratios are not even in training and testing splits.\n\
                         Label ratio in training set: {label_ratio_train}\n\
                        Label ratio in testing set: {label_ratio_test}")
    else:
        return X_train, X_test, y_train, y_test