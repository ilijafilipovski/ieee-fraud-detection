sns.boxplot(x = transTrain['card6'], y = np.log(transTrain[transTrain['TransactionAmt']<=450]['TransactionAmt']))

sns.boxplot(x = transTrain['card6'], y = np.log(transTrain[transTrain['isFraud']==1]['TransactionAmt']))

plt.figure(figsize=(16,12))
sns.boxplot(x = transTrain['isFraud'], y = np.log(transTrain['TransactionAmt']),hue = transTrain['card6'])

sns.boxplot(x = transTrain['card6'], y = np.log(transTrain['TransactionAmt']))

transTrain.groupby(['card6','isFraud','TransactionAmt']).size()
