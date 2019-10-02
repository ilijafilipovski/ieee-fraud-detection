import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import datetime
from sklearn import linear_model
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import cross_val_predict
pd.set_option('display.expand_frame_repr', False)
    
    
def resumeTable(df):
    print('Input dataframe shape is {}'.format(df.shape))
    summary = pd.DataFrame(data = df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary = summary.rename(columns = {'index': 'Name'})
    summary['Missing'] = df.isnull().sum().values
    summary['Missing Percentage'] = (summary['Missing']/df.shape[0])*100
    summary['Unique'] = df.nunique().values
    summary['Unique Percentage'] = (summary['Unique']/df.shape[0])*100
    return summary
    '''
    Funkcija za namaluvaje na iskoristena memorija
    '''
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

''' 
Funckija za naoganje na broj na outliers. Se koristat srednata vrednost na odredena kolona i standardnata devijacija.
'''
def CalcOutliers(col):
    mean = col.mean()
    std = col.std()
    cut = std * 3
    
    lower = mean - cut
    upper = mean + cut
    
    lowerOutliers = [x for x in col if x<lower]
    higherOutliers = [x for x in col if x>upper]
    noOutliers = [x for x in col if x > lower and x < upper]
    
    totalOutliers = [x for x in col if x<lower or x>upper]
    
    print('Identified lowest outliers: %d' % len(lowerOutliers)) 
    print('Identified upper outliers: %d' % len(higherOutliers)) 
    print('Total outlier observations: %d' % len(totalOutliers)) 
    print('Non-outlier observations: %d' % len(noOutliers)) 
    print("Total percentual of Outliers: ", round((len(totalOutliers) / len(noOutliers) )*100, 4))

''' 
Funkcija za kreiranje na crosstab od 2 varijabli

'''    
def createCrosstab(col1, col2):
    crosstab = pd.crosstab(col1, col2, normalize='index')*100
    crosstab = crosstab.reset_index()
    crosstab.rename(columns= {0:'No Fraud', 1: 'Fraud'}, inplace = True)
    return crosstab

def confusion_matrix(col1, col2):
    matrix = pd.crosstab(col1, col2).as_matrix()
    return matrix

def dtypeSeparation(df, num, cat):
    for col in df:
        if df[col].dtype == 'O':
            cat.append(col)
        numerics = ['float16', 'float32', 'int32', 'int16', 'int8', 'float64']
        for element in numerics:
            if df[col].dtype == element:
                num.append(col)

def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def dropHighlyCorrelatedValues(df):

    corr_matrix = df.corr()    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.70)]
    df.drop(df[to_drop], axis=1, inplace = True)

def createCountplotWithTarget(df ,col1, col2):             
    tmp = createCrosstab(df[col1], df[col2])
    g = sns.countplot(df[col1], order = list(tmp[col1].values))
    g.set_title("{} Distribution\n Count and %Fraud by each category".format(col1), fontsize=18)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
        height + 3,
        '{:1.2f}%'.format(height/total*100),
        ha="center", fontsize=13)
    g1 = g.twinx()
    g1 = sns.pointplot(x = col1, y = 'Fraud', data = tmp, color = 'black',  order = list(tmp[col1].values))
    g1.set_ylim(0,(tmp['Fraud'].max())*1.1)
    g1.set_ylabel('% of fraudalent transactions', fontsize = 14)
    plt.show()

def distributionByTarget(df, col1, col2):
    #col2 is target
    g = sns.distplot(df[df[col2] == 1][col1], label='Fraud')
    g = sns.distplot(df[df[col2] == 0][col1], label='NoFraud')
    g.legend()
    g.set_title("{} Distribution\n Count and %Fraud by each category".format(col1), fontsize=18)
    g.set_xlabel("{} Values".format(col1), fontsize=18)
    g.set_ylabel("Probability", fontsize=18)
    plt.show()    

def replaceMissingValues(df, keyword):
    for col in df.columns:
        if keyword in col:
            if df[col].dtype == 'O':
                df[col] = df[col].fillna('Miss')
            else:
                df[col] = df[col].fillna(0)  

def replaceEmails(df, col):        
    df.loc[df[col].isin(['gmail.com', 'gmail']), col] = 'Google'
    df.loc[df[col].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk','yahoo.co.jp', 'yahoo.de', 'yahoo.fr','yahoo.es']), col] = 'Yahoo'
    df.loc[df[col].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 'hotmail.es','hotmail.co.uk', 'hotmail.de', 'outlook.es', 'live.com', 'live.fr', 'hotmail.fr']), col] = 'Microsoft'
    df.loc[df[col].isin(df[col].value_counts()[df[col].value_counts() <= 500 ].index), col] = "Others"    

def compute_roc_auc(index, X, y):
    y_predict = clf.predict_proba(X.iloc[index])[:,1]
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

def plot_roc_curve(fprs, tprs):
    """Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates."""
    
    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(14,10))
    
    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
    # Plot the luck line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    
    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.show()
    return (f, ax)

transTest = pd.read_csv('test_transaction.csv')
transTrain = pd.read_csv('train_transaction.csv')
idTest = pd.read_csv('test_identity.csv')
idTrain = pd.read_csv('train_identity.csv')

transTrain = reduce_mem_usage(transTrain)
transTest = reduce_mem_usage(transTest)
idTrain = reduce_mem_usage(idTrain)
idTest = reduce_mem_usage(idTest)


resume = resumeTable(transTrain)
transTrain['isFraud'] = transTrain['isFraud'].astype('object')

for col in transTrain.columns:
    replaceMissingValues(transTrain, col)
    
for col in transTest.columns:
    replaceMissingValues(transTest, col)
    
numTransTrain, catTransTrain = [], []
dtypeSeparation(transTrain, numTransTrain, catTransTrain)

numTransTest, catTransTest = [], []
dtypeSeparation(transTrain, numTransTest, catTransTest)

transTrainCategorical = pd.DataFrame(transTrain[catTransTrain], columns=catTransTrain)
transTrainNumerical = pd.DataFrame(transTrain[numTransTrain], columns=numTransTrain)

transTestCategorical = pd.DataFrame(transTrain[catTransTrain], columns=catTransTest)
transTestNumerical = pd.DataFrame(transTrain[numTransTest], columns=numTransTest)


correlationCategoricalTarget = dict()
for col in transTrainCategorical.columns:
    try:
        confususionMatrix = confusion_matrix(transTrainCategorical[col], transTrainCategorical['isFraud'])
        correlation = cramers_v(confususionMatrix)
        correlationCategoricalTarget.update({col: correlation })
    except:
        next

correlationNumericalTarget= dict()
for col in transTrainNumerical.columns:
    try:
        confususionMatrix = confusion_matrix(transTrainNumerical[col], transTrainCategorical['isFraud'])
        correlation = cramers_v(confususionMatrix)
        correlationNumericalTarget.update({col: correlation })
    except:
        next

correlationNumericalTarget = pd.DataFrame.from_dict(correlationNumericalTarget, orient='index')
correlationNumericalTarget = correlationNumericalTarget.reset_index()
correlationNumericalTarget.rename(columns = {'index': 'Variable', 0: 'CorrelationValue'}, inplace = True)

numericalCorr = transTrainNumerical.corr().abs()
np.fill_diagonal(numericalCorr.values, -2)
s = numericalCorr.unstack()
numericalCorrelatedPairs = s.sort_values(kind="quicksort")
numericalCorrUpper = numericalCorr.where(np.triu(np.ones(numericalCorr.shape), k=1).astype(np.bool))
numericalToDrop = [column for column in numericalCorrUpper.columns if any(numericalCorrUpper[column] > 0.95)]
numericalCorrelatedPairs = numericalCorrelatedPairs.to_frame()
indexNames = numericalCorrelatedPairs[numericalCorrelatedPairs[0] == -2 ].index
numericalCorrelatedPairs.drop(indexNames , inplace=True)
numericalCorrelatedPairs = numericalCorrelatedPairs.reset_index()
numericalCorrelatedPairs.drop(0, axis = 0, inplace = True)
numericalCorrelatedPairs.iloc[1::2]

transTrain['TransactionAmt'] = transTrain['TransactionAmt'].astype(float)
transTest['TransactionAmt'] = transTrain['TransactionAmt'].astype(float)
total = float(len(transTrain))
plt.figure()
ax = sns.countplot(x= 'isFraud', data=transTrain)
ax.set_xlabel('isFraud?')
ax.set_ylabel('Count')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
    height + 3,
    '{:1.2f}%'.format((height/total)*100),
    ha="center") 
plt.show()

print('TransactionAmt quantiles:')
print(transTrain['TransactionAmt'].quantile([0,0.25, 0.5, 0.75, 1]))

plt.figure(figsize = (8,10))    
g = sns.distplot(transTrain[transTrain['TransactionAmt'] <= 1000]['TransactionAmt'])
g.set_xlabel('TransactionAmt', fontsize = 15)
plt.show()

plt.figure()
g = sns.distplot(transTrain[(transTrain['isFraud'] == 1) & (transTrain.TransactionAmt <1000)]['TransactionAmt'], label = 'Fraud')
g = sns.distplot(transTrain[(transTrain['isFraud'] == 0) & (transTrain.TransactionAmt <1000)]['TransactionAmt'], label = 'No Fraud')
g.set(xlim = (0.001))
g.legend()
plt.show()


plt.figure()
g = sns.distplot(np.log(transTrain[transTrain['isFraud'] == 1]['TransactionAmt']), label = 'Fraud')
g = sns.distplot(np.log(transTrain[transTrain['isFraud'] == 0]['TransactionAmt']), label = 'No Fraud')
g.set(xlim = (1))
g.legend()
plt.show()


CalcOutliers(transTrain.TransactionAmt)
TransactionAmtMean = transTrain['TransactionAmt'].mean()
TransactionAmtMeanFraud = transTrain[transTrain['isFraud'] == 1]['TransactionAmt'].mean()
TransactionAmtMeanNoFraud = transTrain[transTrain['isFraud'] == 0]['TransactionAmt'].mean()

createCountplotWithTarget(transTrain, 'ProductCD', 'isFraud')

resumeTable(transTrain[['card1', 'card2', 'card3', 'card4', 'card5', 'card6']])

gcard4 = createCountplotWithTarget(transTrain, 'card4',  'isFraud')
gcard6 = createCountplotWithTarget(transTrain, 'card6', 'isFraud')
gcard1 = distributionByTarget(transTrain, 'card1', 'isFraud')    
gcard2 = distributionByTarget(transTrain, 'card2', 'isFraud')    

plt.figure(figsize=(8,22))
plt.subplot(413)
g = sns.distplot(transTrain[transTrain['isFraud'] == 0]['card3'], label = 'No Fraud')
g = sns.distplot(transTrain[transTrain['isFraud'] == 1]['card3'], label = 'Fraud')
g.legend()
g.set(xlim = (140, 200))

plt.figure(figsize=(8,22))
plt.subplot(414)
g = sns.distplot(transTrain[transTrain['isFraud'] == 0]['card5'], label = 'No Fraud')
g = sns.distplot(transTrain[transTrain['isFraud'] == 1]['card5'], label = 'Fraud')
g.legend()
g.set(xlim = (90, 240))

resumeTable(transTrain[['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']])

for col in transTrain.columns:
    if 'M' in col:
        plt.figure()
        createCountplotWithTarget(transTrain, col, 'isFraud')
        plt.show()

resumeTable(transTrain[['addr1', 'addr2']])        
ga1 = distributionByTarget(transTrain, 'addr1', 'isFraud')
ga2 = distributionByTarget(transTrain, 'addr2', 'isFraud')

resumeTable(transTrain[['P_emaildomain']])
replaceEmails(transTrain, 'P_emaildomain')
replaceEmails(transTest, 'P_emaildomain')


plt.figure(figsize=(28,14))
gp = createCountplotWithTarget(transTrain, 'P_emaildomain', 'isFraud')

resumeTable(transTrain[['R_emaildomain']])
replaceEmails(transTrain, 'R_emaildomain')
replaceEmails(transTest, 'R_emaildomain')
plt.figure(figsize=(40,20))
gp = createCountplotWithTarget(transTrain, 'R_emaildomain', 'isFraud')

resumeTable(transTrain[['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']])

transTrain.loc[transTrain.C1.isin(transTrain.C1.value_counts()[transTrain.C1.value_counts() <= 400 ].index), 'C1'] = "Others"
transTest.loc[transTest.C1.isin(transTest.C1.value_counts()[transTest.C1.value_counts() <= 400 ].index), 'C1'] = "Others"

plt.figure(figsize = (15,8))
gc1 = createCountplotWithTarget(transTrain, 'C1', 'isFraud')

transTrain.loc[transTrain.C2.isin(transTrain.C2.value_counts()[transTrain.C2.value_counts() <= 350 ].index), 'C2'] = "Others"
transTest.loc[transTest.C2.isin(transTest.C2.value_counts()[transTest.C2.value_counts() <= 350 ].index), 'C2'] = "Others"
plt.figure(figsize = (10,10))
gc2 = createCountplotWithTarget(transTrain, 'C2', 'isFraud')

transTrain.loc[transTrain.C3.isin(transTrain.C3.value_counts()[transTrain.C3.value_counts() <= 2000 ].index), 'C3'] = "Others"    
transTest.loc[transTest.C3.isin(transTest.C3.value_counts()[transTest.C3.value_counts() <= 2000 ].index), 'C3'] = "Others"    
gc3 = createCountplotWithTarget(transTrain, 'C3', 'isFraud')

transTrain.loc[transTrain.C4.isin(transTrain.C4.value_counts()[transTrain.C4.value_counts() <= 350 ].index), 'C4'] = "Others"    
transTest.loc[transTest.C4.isin(transTest.C4.value_counts()[transTest.C4.value_counts() <= 350 ].index), 'C4'] = "Others"    
gc4 = createCountplotWithTarget(transTrain, 'C4', 'isFraud')

transTrain.loc[transTrain.C5.isin(transTrain.C5.value_counts()[transTrain.C5.value_counts() <= 400 ].index), 'C5'] = "Others"
transTest.loc[transTest.C5.isin(transTest.C5.value_counts()[transTest.C5.value_counts() <= 400 ].index), 'C5'] = "Others"
plt.figure(figsize = (10, 8))
gc5 = createCountplotWithTarget(transTrain, 'C5', 'isFraud')

transTrain.loc[transTrain.C6.isin(transTrain.C6.value_counts()[transTrain.C6.value_counts() <= 700 ].index), 'C6'] = "Others"
transTest.loc[transTest.C6.isin(transTest.C6.value_counts()[transTest.C6.value_counts() <= 700 ].index), 'C6'] = "Others"
gc6 = createCountplotWithTarget(transTrain, 'C6', 'isFraud')

transTrain.loc[transTrain.C7.isin(transTrain.C7.value_counts()[transTrain.C7.value_counts() <= 1000 ].index), 'C7'] = "Others"
transTest.loc[transTest.C7.isin(transTest.C7.value_counts()[transTest.C7.value_counts() <= 1000 ].index), 'C7'] = "Others"
gc7 = createCountplotWithTarget(transTrain, 'C7', 'isFraud')

transTrain.loc[transTrain.C8.isin(transTrain.C8.value_counts()[transTrain.C8.value_counts() <= 100 ].index), 'C8'] = "Others"
transTest.loc[transTest.C8.isin(transTest.C8.value_counts()[transTest.C8.value_counts() <= 100 ].index), 'C8'] = "Others"
plt.figure(figsize = (8,18))
gc8 = createCountplotWithTarget(transTrain, 'C8', 'isFraud')

transTrain.loc[transTrain.C9.isin(transTrain.C9.value_counts()[transTrain.C9.value_counts() <= 500 ].index), 'C9'] = "Others"
transTest.loc[transTest.C9.isin(transTest.C9.value_counts()[transTest.C9.value_counts() <= 500 ].index), 'C9'] = "Others"
plt.figure(figsize = (8,18))
gc9 = createCountplotWithTarget(transTrain, 'C9', 'isFraud')

transTrain.loc[transTrain.C10.isin(transTrain.C10.value_counts()[transTrain.C10.value_counts() <= 500 ].index), 'C10'] = "Others"
transTest.loc[transTest.C10.isin(transTest.C10.value_counts()[transTest.C10.value_counts() <= 500 ].index), 'C10'] = "Others"
#plt.figure(figsize = (8,18))
gc10 = createCountplotWithTarget(transTrain, 'C10', 'isFraud')

transTrain.loc[transTrain.C11.isin(transTrain.C11.value_counts()[transTrain.C11.value_counts() <= 500 ].index), 'C11'] = "Others"
transTest.loc[transTest.C11.isin(transTest.C11.value_counts()[transTest.C11.value_counts() <= 500 ].index), 'C11'] = "Others"
gc11 = createCountplotWithTarget(transTrain, 'C11', 'isFraud')

transTrain.loc[transTrain.C12.isin(transTrain.C12.value_counts()[transTrain.C12.value_counts() <= 600 ].index), 'C12'] = "Others"
transTest.loc[transTest.C12.isin(transTest.C12.value_counts()[transTest.C12.value_counts() <= 600 ].index), 'C12'] = "Others"
gc12 = createCountplotWithTarget(transTrain, 'C12', 'isFraud')

transTrain.loc[transTrain.C13.isin(transTrain.C13.value_counts()[transTrain.C13.value_counts() <= 2500 ].index), 'C13'] = "Others"
transTest.loc[transTest.C13.isin(transTest.C13.value_counts()[transTest.C13.value_counts() <= 2500 ].index), 'C13'] = "Others"
gc13 = createCountplotWithTarget(transTrain, 'C13', 'isFraud')

transTrain.loc[transTrain.C14.isin(transTrain.C14.value_counts()[transTrain.C14.value_counts() <= 350 ].index), 'C14'] = "Others"
transTest.loc[transTest.C14.isin(transTest.C14.value_counts()[transTest.C14.value_counts() <= 350 ].index), 'C14'] = "Others"
gc14 = createCountplotWithTarget(transTrain, 'C14', 'isFraud')

START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
transTrain['Date'] = transTrain['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
transTest['Date'] = transTest['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))


transTrain['Weekdays'] = transTrain['Date'].dt.dayofweek
transTrain['Hours'] = transTrain['Date'].dt.hour
transTrain['Days'] = transTrain['Date'].dt.day

transTest['Weekdays'] = transTest['Date'].dt.dayofweek
transTest['Hours'] = transTest['Date'].dt.hour
transTest['Days'] = transTest['Date'].dt.day

transTrain.drop('Date', axis = 1, inplace = True)
transTest.drop('Date', axis = 1, inplace = True)


gdays = createCountplotWithTarget(transTrain, 'Days', 'isFraud')
gweekdays = createCountplotWithTarget(transTrain, 'Weekdays', 'isFraud')
ghours = createCountplotWithTarget(transTrain, 'Hours', 'isFraud')

numIdTrain, catIdTrain = [], []
numIdTest, catIdTest = [], []
dtypeSeparation(idTrain, numIdTrain, catIdTrain)
dtypeSeparation(idTest, numIdTest, catIdTest)

df_train = transTrain.merge(idTrain, how = 'left', left_index=True, right_index=True, on = 'TransactionID')
df_test = transTest.merge(idTest, how = 'left', left_index=True, right_index=True, on = 'TransactionID')    
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)

for col in df_train.columns:
    replaceMissingValues(df_train, col)

for col in df_test.columns:
    replaceMissingValues(df_test, col)
    
numTrain, catTrain = [], []
numTest, catTest = [], []
dtypeSeparation(df_train, numTrain, catTrain)
dtypeSeparation(df_test, numTest, catTest)

for col in catTrain:
    if 'id_' in col:
        plt.figure()
        createCountplotWithTarget(df_train, col, 'isFraud')
        plt.show()
        

    
        
df_train.loc[df_train['id_30'].str.contains('Windows'), 'id_30'] = 'Windows'
df_train.loc[df_train['id_30'].str.contains('iOS'), 'id_30'] = 'iOS'        
df_train.loc[df_train['id_30'].str.contains('Mac OS'), 'id_30'] = 'Mac'        
df_train.loc[df_train['id_30'].str.contains('Android'), 'id_30'] = 'Android'

df_test.loc[df_test['id_30'].str.contains('Windows'), 'id_30'] = 'Windows'
df_test.loc[df_test['id_30'].str.contains('iOS'), 'id_30'] = 'iOS'        
df_test.loc[df_test['id_30'].str.contains('Mac OS'), 'id_30'] = 'Mac'        
df_test.loc[df_test['id_30'].str.contains('Android'), 'id_30'] = 'Android'

plt.figure()
createCountplotWithTarget(df_train, 'id_30', 'isFraud')

df_train.loc[df_train['id_31'].str.contains('chrome'), 'id_31'] = 'Chrome'
df_train.loc[df_train['id_31'].str.contains('firefox'), 'id_31'] = 'Firefox'
df_train.loc[df_train['id_31'].str.contains('samsung'), 'id_31'] = 'Samsung'
df_train.loc[df_train['id_31'].str.contains('safari'), 'id_31'] = 'Safari'
df_train.loc[df_train['id_31'].str.contains('edge'), 'id_31'] = 'Edge'
df_train.loc[df_train['id_31'].str.contains('ie'), 'id_31'] = 'IE'
df_train.loc[df_train['id_31'].str.contains('opera'), 'id_31'] = 'Opera'
df_train.loc[df_train.id_31.isin(df_train.id_31.value_counts()[df_train.id_31.value_counts() < 200].index), 'id_31'] = "Others"

df_test.loc[df_test['id_31'].str.contains('chrome'), 'id_31'] = 'Chrome'
df_test.loc[df_test['id_31'].str.contains('firefox'), 'id_31'] = 'Firefox'
df_test.loc[df_test['id_31'].str.contains('samsung'), 'id_31'] = 'Samsung'
df_test.loc[df_test['id_31'].str.contains('safari'), 'id_31'] = 'Safari'
df_test.loc[df_test['id_31'].str.contains('edge'), 'id_31'] = 'Edge'
df_test.loc[df_test['id_31'].str.contains('ie'), 'id_31'] = 'IE'
df_test.loc[df_test['id_31'].str.contains('opera'), 'id_31'] = 'Opera'
df_test.loc[df_test.id_31.isin(df_test.id_31.value_counts()[df_test.id_31.value_counts() < 200].index), 'id_31'] = "Others"

createCountplotWithTarget(df_train, 'id_31', 'isFraud')

for col in numTrain:
    if 'id_' in col:
        plt.figure()
        distributionByTarget(df_train, col, 'isFraud')
        plt.show()

df_train.drop('id_33', axis = 1, inplace = True)
df_test.drop('id_33', axis = 1, inplace = True)

#Modelling
        
for f in df_train.drop('isFraud', axis=1).columns:
    if df_train[f].dtype == 'O' or df_test[f].dtype == 'O' or df_train[f].dtype == 'object' or df_test[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_train[f].values) + list(df_test[f].values))
        df_train[f] = lbl.transform(list(df_train[f].values))
        df_test[f] = lbl.transform(list(df_test[f].values))

df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)


X = df_train.drop('isFraud', axis = 1)
y = df_train['isFraud']
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# =============================================================================
# scores = []
# best_svr = SVR(kernel='rbf')
# cv = KFold(n_splits=10, random_state=42, shuffle=False)
# for train_index, test_index in cv.split(X):
#     print("Train Index: ", train_index, "\n")
#     print("Test Index: ", test_index)
# 
#     X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
#     best_svr.fit(X_train, y_train)
#     scores.append(best_svr.score(X_test, y_test))
#         
# =============================================================================

svr = SVR(kernel='rbf')
clf = RandomForestClassifier(
    n_estimators=50,
    criterion='gini',
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    random_state=0,
    verbose=0,
    warm_start=False,
    class_weight='balanced'
)
y_pred = cross_val_predict(clf, X, y, cv=10)
y_pred1 = cross_val_predict(svr, X,y, cv = 5)
report = classification_report(y, y_pred, output_dict=True)
dfReport = pd.DataFrame(report)
