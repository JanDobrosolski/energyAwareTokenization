import os
import pandas as pd
import copy
from tqdm import tqdm

from tokenizers.pre_tokenizers import Whitespace

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
encodeString = ''
totalLength = len(fullDataset)
print("dataset loaded, time to split it")

cond1 = totalLength//4
cond2 = totalLength//16

whiteSpacePretokenizer = Whitespace()
i = 0
for id, row in tqdm(fullDataset[:cond1].iterrows(), total=totalLength//4):
    tokenisedString = whiteSpacePretokenizer.pre_tokenize_str(row['Title']+ " " + row['Review'])
    tokenisedString = [x[0] for x in tokenisedString]
    finalString += ' '.join(tokenisedString)+' '
    if i == cond2:
        encodeString = copy.deepcopy(finalString)
    i += 1

print("saving the dataset 1")
with open('processedDatasetTrainSmall.txt', 'w') as f:
    f.write(finalString)

print("saving the dataset 2")
with open('processedDatasetTestSmall.txt', 'w') as f:
    f.write(encodeString)

