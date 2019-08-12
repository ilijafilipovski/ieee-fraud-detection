    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import scipy as sp
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as ss
    
    
    transTest = pd.read_csv('test_transaction.csv')
    transTrain = pd.read_csv('train_transaction.csv')
    idTest = pd.read_csv('test_identity.csv')
    idTrain = pd.read_csv('train_identity.csv')
    
    
    '''
    Funkcija za rezime na odreden dataset. 
    Funkcijata vraka dataframe vo koj se sodrzat informacii za varijablite na datasetot koj e vnesen kako argument
    
    '''
    
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
        noOutliers = [x for x in col if x>lower and x<upper]
        
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
    
    def dtypeSeparation(df):
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
    
    
    # =============================================================================
    # train = pd.merge(transTrain, idTrain, on = 'TransactionID', how = 'left')
    # test = pd.merge(transTest, idTest, on = 'TransactionID', how = 'left')
    # =============================================================================
    
    transTrain = reduce_mem_usage(transTrain)
    idTrain = reduce_mem_usage(idTrain)
    
    resume = resumeTable(transTrain)
    print("transTrain has {} rows and {} columns".format(transTrain.shape[0], transTrain.shape[1]))
    
    transTrain['isFraud'] = transTrain['isFraud'].astype('object')
    
    
    '''
        Funckija za podelba na varijablite od odreden dataset na numericki i kategoricki,
        odnosno stavanje na nivnite iminja vo odredena lista koja sto ke se iskoristi ponatamu(se misli na listata)
    '''
    cat, num = [], []
    dtypeSeparation(transTrain)
    
    '''
        Kreiranje na dataframe koj sto se sodrzi samo so kategorickite varijabli
    '''
    
    transTrainCategorical = pd.DataFrame(transTrain[cat], columns=cat)
    
    '''
        Kreiranje na dataframe koj sto se sodrzi samo so numericki varijabli
    '''
    
    transTrainNumerical = pd.DataFrame(transTrain[num], columns=num)
    
    
    
    # =============================================================================
    # transTrain.drop(columns= 'P_emaildomain', axis = 1, inplace = True)
    # transTrainCategorical.drop(columns= 'P_emaildomain', axis = 1, inplace = True)
    # 
    # transTrain.drop(columns= 'TransactionID', axis = 1, inplace = True)
    # transTrainNumerical.drop(columns= 'TransactionID', axis = 1, inplace = True)
    # =============================================================================
    
    
    
    '''
        Kreiranje na dictionaries vo koj se kako key e zacuvana varijablata, a kako value korelacijata koja taa
        varijabla ja ima so target varijablata(isFraud)    
    '''
    
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
    # =============================================================================
    #
    # highlyCorrelatedCategorical = dict()
    # for (key, value) in correlationCategoricalTarget.items():
    #    if value > 0.1:
    #        highlyCorrelatedCategorical[key] = value
    #  
    # print('Highly Correlated Categorical variables: ')
    # print(highlyCorrelatedCategorical)
    #         
    # highlyCorrelatedNumerical = dict()
    # for (key, value) in correlationNumerical.items():
    #   if value > 0.5:
    #      highlyCorrelatedNumerical[key] = value
    # 
    # print('Highly Correlated Numerical variables: ')
    # print(highlyCorrelatedNumerical) 
    #    
    # =============================================================================
    
    # =============================================================================
    # noMissingValues = list(resume[resume['Missing'] == 0]['Name'])
    # lowMissingValues = list(resume[resume['Missing Percentage'] <= 52]['Name'])
    # highMissingValues = list(resume[resume['Missing Percentage'] > 52]['Name'])
    # =============================================================================
    
    transTrainNumerical = transTrainNumerical.dropna()
    
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
    #----------------------------------------------------------------------------------------
    '''
        Distribucija na target varijabla.
    '''
    
    transTrain['TransactionAmt'] = transTrain['TransactionAmt'].astype(float)
    total = float(len(transTrain))
    ax = sns.countplot(x= 'isFraud', data=transTrain)
    ax.set_xlabel('isFraud?')
    ax.set_ylabel('Count')
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format((height/total)*100),
                ha="center") 
    
    
    
    '''
        Presmetka na kvantilite na TransactionAmt varijablata od trening mnozestvoto
    '''
    
    print('TransactionAmt quantiles:')
    print(transTrain['TransactionAmt'].quantile([0,0.25, 0.5, 0.75, 1]))
    
    
    '''
        Distribucija na TransactionAmt varijablata. Prikazanata distribucija i na logaritmirani vrednosti
    '''
    
    plt.figure(figsize = (8,10))    
    g = sns.distplot(transTrain[transTrain['TransactionAmt'] <= 1000]['TransactionAmt'])
    g.set_xlabel('TransactionAmt', fontsize = 15)
    plt.show()
    
    
    g = sns.distplot(np.log(transTrain[transTrain['isFraud'] == 1]['TransactionAmt']), label = 'Fraud')
    g = sns.distplot(np.log(transTrain[transTrain['isFraud'] == 0]['TransactionAmt']), label = 'No Fraud')
    g.set(xlim = (1))
    g.legend()
    plt.show()
    
    g = sns.distplot(transTrain[(transTrain['isFraud'] == 1) & (transTrain.TransactionAmt <1000)]['TransactionAmt'], label = 'Fraud')
    g = sns.distplot(transTrain[(transTrain['isFraud'] == 0) & (transTrain.TransactionAmt <1000)]['TransactionAmt'], label = 'No Fraud')
    g.set(xlim = (0.001))
    g.legend()
    plt.show()
    
    '''
        Povikana funckcija za naoganje outliers
    '''
    
    CalcOutliers(transTrain.TransactionAmt)
    TransactionAmtMean = transTrain['TransactionAmt'].mean()
    TransactionAmtMeanFraud = transTrain[transTrain['isFraud'] == 1]['TransactionAmt'].mean()
    TransactionAmtMeanNoFraud = transTrain[transTrain['isFraud'] == 0]['TransactionAmt'].mean()
    
    
    resumeTable(transTrain[['card1', 'card2', 'card3', 'card4', 'card5', 'card6']])
    
    
    tmp1 = createCrosstab(transTrain['ProductCD'], transTrain['isFraud'])
    
    g = sns.countplot(x = 'ProductCD', data = transTrain)
    g.set_xlabel('ProductCD', fontsize = 13)
    g.set_ylabel('ProductCD Counts', fontsize=13)
    g.set_title('ProductCD value distribution', fontsize = 13)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=13)
    plt.show()
        
    g1 = sns.countplot(x = 'ProductCD', hue='isFraud', data = transTrain)
    g1.set_xlabel('isFraud by ProductCD', fontsize = 13)
    g1.set_ylabel('isFraud by ProductCD Counts', fontsize=13)
    g1.set_title('isFraud by ProductCD value distribution', fontsize = 13)
    plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])
    gt = g1.twinx()
    gt = sns.pointplot(x = 'ProductCD', y = 'Fraud', data = tmp1, order=['W', 'H', 'C', 'S', 'R'], color= 'red')
    plt.show()
    
    
    tmp2 = createCrosstab(transTrain['card4'], transTrain['isFraud'])
    
    g = sns.countplot(x = 'card4', data = transTrain)
    g.set_xlabel('Card 4', fontsize = 14)
    g.set_ylabel('Card 4 count', fontsize = 14)
    g.set_title('Card 4 count distribution', fontsize = 14)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=14)
    plt.show()
    
    g1 = sns.countplot(x = 'card4', hue='isFraud', data = transTrain)
    g1.set_xlabel('Card 4', fontsize = 14)
    g1.set_ylabel('Card 4 count', fontsize = 14)
    g1.set_title('Card 4 count distribution', fontsize = 14)
    gt = g1.twinx()
    gt = sns.pointplot(x = 'card4', y = 'Fraud' , data = tmp2, order=['discover', 'mastercard', 'visa', 'american express'], color = 'black')
    plt.show()
    
    g1 = sns.countplot(x = transTrain[transTrain['isFraud'] == 1]['card4'])
    g1.set_xlabel('Card 4', fontsize = 14)
    g1.set_ylabel('Card 4 count', fontsize = 14)
    g1.set_title('Card 4 count distribution', fontsize = 14)
    gt = g1.twinx()
    gt = sns.pointplot(x = 'card4', y = 'Fraud' , data = tmp2, order=['discover', 'mastercard', 'visa', 'american express'], color = 'black')
    plt.show()
    
    
    tmp3 = createCrosstab(transTrain['card6'], transTrain['isFraud'])
    
    plt.figure(figsize=(8,8))
    g = sns.countplot(x = 'card6', data = transTrain)
    g.set_title('Card 6 Countplot', fontsize = 14)
    g.set_xlabel('Card 6 Values', fontsize = 14)
    g.set_ylabel('Card 6 Count', fontsize = 14)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=14)
    gt = g.twinx()
    gt = sns.pointplot(x = 'card6', y = 'Fraud',  data = tmp3, order=['credit', 'debit', 'debit or credit', 'charge card'], color = 'black')
    plt.show()
    
    plt.figure(figsize=(8,8))
    g1 = sns.countplot(x = 'card6', hue = 'isFraud', data = transTrain)
    plt.show()
    
    tmp4 = createCrosstab(transTrain['card1'], transTrain['isFraud'])
    
    plt.figure(figsize=(8,22))
    plt.subplot(411)
    g = sns.distplot(transTrain[transTrain['isFraud'] == 1]['card1'], label='Fraud')
    g = sns.distplot(transTrain[transTrain['isFraud'] == 0]['card1'], label='NoFraud')
    g.legend()
    g.set_title("Card 1 Values Distribution by Target", fontsize=20)
    g.set_xlabel("Card 1 Values", fontsize=18)
    g.set_ylabel("Probability", fontsize=18)
    plt.show()