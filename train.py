import time
import os
from argparse import ArgumentParser

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, \
                                WordPieceTrainer, UnigramTrainer

UNK_TOKENS = "<UNK>"
SPL_TOKENS = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]

# PROCESSED_FILE = "processedDataset.txt"
TRAINING_FILE = "/home/macierz/s175327/tokenizationEnergy/processedDatasetTrainSmall.txt"
TESTING_FILE = "/home/macierz/s175327/tokenizationEnergy/processedDatasetTestSmall.txt"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tokenizerType", type=str, required=False, default="BPE",choices=['BPE', 'UNI', 'WPC', 'WPL'],
                        help="tokenizer type")
    
    args = parser.parse_args()

    tokenizerType = args.tokenizerType
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

    start = time.time()
    tokenizer.train_from_iterator(dataset, trainer, len(dataset))
    end = time.time()
    os.makedirs(f"tokenizer{tokenizerType}", exist_ok=True)
    tokenizer.save(os.path.join(f"tokenizer{tokenizerType}",f"tokenizer{tokenizerType}.json"))
    print(f"Training took {end - start} seconds")

    # with open('processedDataset.txt', 'r') as f:

    # tokenizer = Tokenizer.from_file(os.path.join(f"tokenizer{tokenizerType}",f"tokenizer{tokenizerType}.json"))
    # start = time.time()
    # encoded = tokenizer.encode(dataset, is_pretokenized=True)
    # end = time.time()
    # print(f"Encoding took {end - start} seconds")
    
