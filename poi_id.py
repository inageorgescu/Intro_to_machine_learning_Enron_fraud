#!/usr/bin/python


import sys
sys.path.append("../tools/")
import pickle

import numpy as np
import pandas as pd

#from operator import itemgetter
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectKBest

import sklearn.feature_selection
import sklearn.pipeline
import sklearn.model_selection
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
sns.set_style('darkgrid')
plt.style.use('ggplot')

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                  'from_messages', 
                  'to_messages',
                  'from_poi_to_this_person', 
                  'from_this_person_to_poi', 
                  'shared_receipt_with_poi',
                  'salary', 
                  'bonus', 
                  'total_payments',
                  'total_stock_value',
                  'deferral_payments', 
                  'deferred_income', 
                  'director_fees', 
                  'exercised_stock_options', 
                  'expenses', 
                  'loan_advances', 
                  'long_term_incentive', 
                  'restricted_stock', 
                  'restricted_stock_deferred',
                  'email_address',
                  'other']

#Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
### Store to my_dataset for easy export below.
my_dataset = data_dict

print '#### My raw data set and features ####'
print len(my_dataset)
#print my_dataset.keys()
#print my_dataset.items()


# Replace NaN  with numpy.nan
for name, d in my_dataset.items():
    for feature, value in d.items():
        if value == 'NaN':
            d[feature] = np.nan

# Convert dictionary of dictionaries into pandas.DataFrame
enrondf = pd.concat(objs=[pd.DataFrame.from_dict(data={k:d}, orient='index') for k, d in my_dataset.items()])
print enrondf.info()

### Task 2: Remove outliers
# Plot scatterplot to find outlierts
grid = sns.JointGrid(enrondf['salary'], enrondf['bonus'], space=0, size=5, ratio=25)
grid.plot_joint(plt.scatter, color="b")
plt.show()

#Examine column names visualy and drop outliers that are not a name of a person
column_names = list(enrondf.index.values)
print'##### Names of POI to be checked for outliers#####'
print column_names

def percent_missing(df):
    """Calculates percentage of missing columns for each record (row)."""
    for index, row in df.iterrows():
        df.ix[index, 'percent_missing'] = (row.size-row.count())*100.0/row.size  
percent_missing(enrondf)

#Print records with more missing values
print 'Percent of missing values'
print enrondf['percent_missing'].sort_values(ascending=False).head()

#Drop outliers
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
enrondf = enrondf.drop(outliers)


### Task 3: Create new feature(s)
# fraction_to_poi = from_this_person_to_poi/from_messages

enrondf['from_to_poi'] = enrondf['from_poi_to_this_person']*1.0 + enrondf['from_this_person_to_poi']*1.0

# fraction_to_poi = from_this_person_to_poi/to_messages
enrondf['fraction_to_poi'] = enrondf['from_this_person_to_poi']*1.0/enrondf['from_messages']
plot1 = sns.JointGrid(enrondf['from_this_person_to_poi'], enrondf['from_messages'], space=0, size=5, ratio=25)
plot1.plot_joint(plt.scatter, color="b")
plt.show()

# fraction_from_poi = from_poi_to_this_person/to_messages
enrondf['fraction_from_poi'] = enrondf['from_poi_to_this_person']*1.0/enrondf['to_messages']

#feature list for grading
my_feature_list = features_list + ['from_to_poi'] + ['fraction_to_poi'] + ['fraction_from_poi']

#Drop columns with too many missing values
columns_to_delete = ['deferral_payments', 'restricted_stock_deferred',
                     'loan_advances', 'director_fees', 'from_poi_to_this_person', 'from_this_person_to_poi', 'email_address']

enrondf = enrondf.drop(labels=columns_to_delete, axis=1)

print'DATAFRAME AFTER FEATURES DELETION/ADDITION'
print enrondf.info()

for col in columns_to_delete:
    my_feature_list.remove(col)

percent_missing(enrondf)

#Drop rows with too many missing values
max_data_percent = 70.0

print 'ROWS TO BE DELETED WITH TOO MANY MISSING VALUES'
print enrondf[enrondf['percent_missing'] > max_data_percent].index
enrondf = enrondf[enrondf['percent_missing'] <= max_data_percent]

print'DATAFRAME INFO AFTER ROW CLEANING'
print enrondf.info()

enrondf = enrondf.drop('percent_missing', axis=1)

#Fill missing values

for feature in my_feature_list[1:]:
    enrondf[feature].fillna(value=enrondf[feature].mean(), inplace=True)
    
#Scale values
enrondf[my_feature_list[1:]] = scale(enrondf[my_feature_list[1:]].values)

#Plot scatterplot to see how data looks after scaling and filling missing values
grid = sns.JointGrid(enrondf['salary'], enrondf['bonus'], space=0, size=5, ratio=25)
grid.plot_joint(plt.scatter, color="b")
plt.show()

#Save to excel
enrondf.to_excel('my_dataframe.xlsx') 
print enrondf.head()

#Select best k features
k_best = SelectKBest()
k_best.fit(enrondf[my_feature_list[1:]].values, enrondf['poi'])
scores = k_best.scores_
unsorted_pairs = zip(my_feature_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[-1])))

print sorted_pairs
dataframe = pd.DataFrame(sorted_pairs, columns=['features', 'kscores'])
dataframe.head()
plot= sns.barplot(dataframe['features'], dataframe['kscores'])
for item in plot.get_xticklabels():
    item.set_rotation(90)
plt.show()

print 'BEST FEATURES'

for feature, score in sorted_pairs:
    print '{:4.2f} | {}'.format(score, feature)

 
# Convert back from pandas.DataFrame to dictionary of dictionaries
my_dataset = {}
for row in enrondf.itertuples():
    d = row._asdict()
    name = d.pop('Index')
    my_dataset[name] = d   

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# more about tune_and_eval_clf at https://www.civisanalytics.com/blog/workflows-in-python-using-pipeline-and-gridsearchcv-for-more-compact-and-comprehensive-code/

def tune_and_eval_clf(clf, params, features, labels):
    """ Tunes and evaluates a classifier """
    
    select = sklearn.feature_selection.SelectKBest()
        
#    print clf
    
    steps = [('feature_selection', select), ('clf', clf)]
    
    pipeline = sklearn.pipeline.Pipeline(steps)
    
    parameters = dict(feature_selection__k=[1, 3, 5, 7, 10, 13, 15])
    parameters.update(params)

    features_train, features_test, labels_train, labels_test = sklearn.model_selection.train_test_split(features, labels, test_size=0.3, random_state=42)
    
#    print features_train
#    print labels_train

    
    sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=60)
    
    cv = sklearn.model_selection.GridSearchCV(estimator=pipeline, cv=sss, param_grid=parameters)
   
    cv.fit(features_train, labels_train)
    
    print 'Parameters Tuned:'
    best_parameters = cv.best_params_
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    
    labels_pred = cv.predict(features_test)
    
    target_names = ['NON-POI', 'POI']
    
    result = sklearn.metrics.classification_report( labels_test, labels_pred, target_names=target_names )
    print 'Result tuning:'
    print result
    
    precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(labels_test, labels_pred, labels=target_names, average='binary')
    print 'Performance 1-Run:'
    print 'Precision = {:8.6f} and Recall = {:7.5f}'.format(precision, recall)

    print labels_test
    
    
##Logistic Regression Classifier
print'TRYING LOGISTIC REGRESSION'
lr_clf = LogisticRegression()
lr_params = {'clf__C': [1e-08, 1e-07, 1e-06],
             'clf__tol': [1e-2, 1e-3, 1e-4],
             'clf__penalty': ['l1', 'l2'],
             'clf__random_state': [42, 46, 60]}

tune_and_eval_clf(lr_clf, lr_params, features, labels)


##Decision Tree Classifier
print'TRYING DECISION TREE'
dt_clf = DecisionTreeClassifier()
dt_params = {"clf__min_samples_leaf": [2, 6, 10, 12],
             "clf__min_samples_split": [2, 6, 10, 12],
             "clf__criterion": ["entropy", "gini"],
             "clf__max_depth": [None, 5],
             "clf__random_state": [42, 46, 60]}

tune_and_eval_clf(dt_clf, dt_params, features, labels)


##Support Vector Machine Classifier
print 'TRYING SUPPORT VECTOR MACHINES'
svc_clf = SVC()
svc_params = {'clf__C': [1000], 'clf__gamma': [0.001], 'clf__kernel': ['rbf']}

tune_and_eval_clf(svc_clf, svc_params, features, labels)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
    
def select_and_dump(algorithm):
    """ Select and dump the algorithm selected. """
    
    algorithm = algorithm.strip().replace(' ','').lower()
    
    if algorithm == 'logisticregression': 
     
        my_clf = LogisticRegression(C=1e-08, penalty='l2', random_state=42, tol=0.01)
        number_best_features = 7
    
    elif algorithm == 'decisiontree':
        my_clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_leaf=2, min_samples_split=6,random_state=42)
        number_best_features = 7 

    elif algorithm in ['svc', 'svm', 'supportvectormachines']:
        my_clf = SVC(C=1000, gamma=0.001, kernel='rbf')
        number_best_features = 7 
        
    clf = my_clf
    best_feature_list = [x[0] for x in sorted_pairs[:number_best_features]]
    
    #My Feature List after selecting K Best featues
    my_feature_list = ['poi'] + best_feature_list 
    
    dump_classifier_and_data(clf, my_dataset, my_feature_list)
    
#check the final data set in terms of number of features and entries
print '#### My final data set and features ####'
print len(my_dataset)
#print my_dataset.keys()
print len(my_feature_list)
print my_feature_list

select_and_dump('decisiontree')

#sources: https://github.com/lyvinhhung/Udacity-Data-Analyst-Nanodegree/blob/master/p5%20-%20Identify%20Fraud%20from%20Enron%20Email/scripts/poi_id.py