import time
import os
from argparse import ArgumentParser
import multiprocessing

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, \
                                WordPieceTrainer, UnigramTrainer

UNK_TOKENS = "<UNK>"
SPL_TOKENS = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]

TOKENIZER_TYPE = "BPE"

# PROCESSED_FILE = "processedDataset.txt"
TRAINING_FILE = "/home/macierz/s175327/tokenizationEnergy/processedDatasetTrainSmall.txt"
TESTING_FILE = "/home/macierz/s175327/tokenizationEnergy/processedDatasetTestSmall.txt"

def trainTokenizer(tokenizerType):
    if tokenizerType == 'BPE':
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKENS))
        trainer = BpeTrainer(special_tokens=SPL_TOKENS)
    elif tokenizerType == 'UNI':
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(special_tokens=SPL_TOKENS)
    elif tokenizerType == 'WPC':
        tokenizer = Tokenizer(WordPiece(unk_token=UNK_TOKENS))
        trainer = WordPieceTrainer(special_tokens=SPL_TOKENS)
    elif tokenizerType == 'WPL':
        tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKENS))
        trainer = WordLevelTrainer(special_tokens=SPL_TOKENS)
    
    with open(TESTING_FILE, 'r') as f:
        dataset = f.read()

    dataset = dataset.split(' ')

    tokenizer.train_from_iterator(dataset, trainer, len(dataset))
    tokenizer.save(os.path.join(f"tokenizer{tokenizerType}",f"tokenizer{tokenizerType}.json"))

if __name__ == "__main__":
    if TOKENIZER_TYPE not in ['BPE', 'UNI', 'WPC', 'WPL']:
        exit()
    process1 = multiprocessing.Process(target=trainTokenizer(TOKENIZER_TYPE))
    process2 = multiprocessing.Process(target=trainTokenizer(TOKENIZER_TYPE))

    process1.start()
    process2.start()

    process1.join()
    process2.join()
    