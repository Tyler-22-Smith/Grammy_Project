from sklearn import svm
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.utils import shuffle
from pandas import DataFrame, read_csv
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from scipy.stats import pearsonr

# read in data
df_raw=pd.read_csv(r'C:\Users\Tyler Smith\Documents\GitHub\Grammy_Project\updated_grammy_nominations.csv')

# remove identifying columns
col_names_remove = ["track_album_id", "track_id", "track_album_name", "track_name", "track_album_release_date", 'artist_main', 'track_type']


# droping the identified columns from the dataframe
df = df_raw.drop(col_names_remove, axis=1)
df['track_explicit'] = df['track_explicit'].astype(int)
df['track_album_release_date_precision'] = df['track_album_release_date_precision'].replace({'day': 0, 'year': 1})
df['track_album_release_date_precision'] = df['track_album_release_date_precision'].astype(int)
print(df.columns)

# balancing dataframe
count_winner = df[df['award_status'] == 'winner'].shape[0]
count_nominated = df[df['award_status'] == 'nominee'].shape[0]
min_count = min(count_winner, count_nominated)
try:
    if count_winner > count_nominated:
        df_winner = df[df['award_status'] == 'winner'].sample(min_count, random_state=42)  # random_state for reproducibility
        df_nominated = df[df['award_status'] == 'nominee']
    elif count_nominated > count_winner:
        df_nominated = df[df['award_status'] == 'nominee'].sample(min_count, random_state=42)
        df_winner = df[df['award_status'] == 'winner']
    else:
        # Already balanced
        df_winner = df[df['award_status'] == 'winner']
        df_nominated = df[df['award_status'] == 'nominee']
finally:
    print("Balancing Complete!")

df_balanced = pd.concat([df_winner, df_nominated])
print(f'Number of rows: {df_balanced.shape[0]}')
print(f'Number of columns: {df_balanced.shape[1]}')
print(df_balanced.head(5))
print(df_balanced.columns)

# shuffle the dataset to make sure that the training and test samples are selected randomly, and not according to some order in the file.
df = shuffle(df_balanced)

# separating to training and test samples. First 300 rows are used for testing, and the rest for training. The number of training and test samples need to be adjusted to the size of the dataset.
test=df.iloc[:330,:]
train=df.iloc[330:,:]
# scikit needs two separate lists - one for the features, and another for the labels. The code below separates the file into two lists. The labels are taken from the column titled "Class". That title can be different in different datasets.
train_labels=train['award_status'].tolist()
train=train.drop(['award_status'],axis=1)
train_samples=train.values.tolist()
test_labels=test['award_status'].tolist()
# very important to drop the class variable from the test set. We don't want theclassifier to know that value and use it for classification
test=test.drop(['award_status'],axis=1)
test_samples=test.values.tolist()
# scikit likes numbers. In this example the classes are "class1" and "class2". Scikit does not like strings, so the nominal (string) values are being replaced with numerical values. "class1" is changed to the number 0, and "class2" is changed to the number 1. If there are more than two classes more if statements will be needed. If the classes are already in numbers, these lines are not needed at all.
for i in range (len(train_labels)):
    if train_labels[i]=='R':
        train_labels[i]=0
    if train_labels[i]=='D':
        train_labels[i]=1
for i in range (len(test_labels)):
    if test_labels[i]=='R':
        test_labels[i]=0
    if test_labels[i]=='D':
        test_labels[i]=1

# Zero-R Classifier: Always predict the majority class (winner or nominee)
majority_class = max(set(train_labels), key=train_labels.count)  # Find the most frequent class
zero_r_predictions = [majority_class] * len(test_labels)  # Predict the majority class for all test samples

# Calculate accuracy for Zero-R
zero_r_accuracy = accuracy_score(test_labels, zero_r_predictions)
print(f"Zero-R Accuracy: {zero_r_accuracy}")

# to test a number of classifiers at a time, we use a list of classifiers
classifiers = [
KNeighborsClassifier(3),
DecisionTreeClassifier(),
RandomForestClassifier(),
AdaBoostClassifier(),
GradientBoostingClassifier(),
GaussianNB(),
LinearDiscriminantAnalysis(),
QuadraticDiscriminantAnalysis()
]
for clf in classifiers:
    clf.fit(train_samples, train_labels)
    res=clf.predict(test_samples)
    acc = accuracy_score(test_labels, res)
    print (clf.__class__.__name__+" Accuracy: "+str(acc))

# the code below finds the number of correct predictions, and prints to the screen the classification accuracy
correct_samples=0
for i in range (len(test_labels)):
    if test_labels[i]==res[i]:
        correct_samples=correct_samples+1

print(correct_samples/len(test_labels))






for clf in classifiers:
    clf.fit(train_samples, train_labels)
    res=clf.predict(test_samples)
    acc = accuracy_score(test_labels, res)
    
    correct_samples = 0
    for i in range(len(test_labels)):
        if test_labels[i]==res[i]:
            correct_samples=correct_samples+1
    percent_correct = correct_samples/len(test_labels)
    print(percent_correct)
    r=pearsonr(percent_correct, 0.5)    

    print("{0}: \n - Accuracy: {1} \n Mean Absolute Error: {2} \n Pearsons r: {3}".format(clf.__class__.__name__, str(round(acc, 2)), str(round(percent_correct, 2)), str(r)))