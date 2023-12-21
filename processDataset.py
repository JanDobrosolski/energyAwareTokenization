import os
import pandas as pd

books = pd.read_csv('dataset/Books_rating.csv')
tempAmazonRev = pd.read_csv('dataset/test.csv')
tempAmazonRev.columns = ['Polarity','Title','Review']

tempAmazonRev2 = pd.read_csv('dataset/train.csv')
tempAmazonRev2.columns = ['Polarity','Title','Review']

books.drop(['Id','Title','Price','User_id','profileName','review/helpfulness','review/score','review/time'], axis=1, inplace=True)
books.columns = ['Title','Review']

amazonReview = pd.concat([tempAmazonRev,tempAmazonRev2],axis=0)
amazonReview.drop('Polarity', axis=1, inplace=True)

fullDataset = pd.concat([books,amazonReview],axis=0)
fullDataset.reset_index(drop=True)
fullDataset['Title'].fillna('no title', inplace=True)
fullDataset['Review'].fillna('no review', inplace=True)

finalString = ''

counter = 0 
for id, row in fullDataset.iterrows():
    finalString += row['Title']+row['Review']
    counter += 1
    if counter == 100:
        import pdb; pdb.set_trace()


with open('processedDataset.txt', 'w') as f:
    f.write(finalString)
