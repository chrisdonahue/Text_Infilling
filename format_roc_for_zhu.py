from collections import Counter
import os
import pickle

from nltk.tokenize import word_tokenize
from tqdm import tqdm

VOCAB_SIZE = 10000

print('Loading')
with open('04_07_12_data_train_sentences.pkl', 'rb') as f:
  train_data = pickle.load(f)
with open('04_07_13_data_valid_sentences.pkl', 'rb') as f:
  valid_data = pickle.load(f)

if os.path.exists('vocab.txt'):
  with open('vocab.txt', 'r') as f:
    vocab = f.read().strip().splitlines()
else:
  print('Counting')
  token_counts = Counter()
  for data in [train_data]:
    for doc, _ in tqdm(data):
      title, doc = doc.splitlines()
      doc_tokens = word_tokenize(doc.strip().lower())
      for t in doc_tokens:
        token_counts[t] += 1

  print('Aggregating')
  vocab = ['<m>', '<BOA>', '<EOA>']
  for token, _ in sorted(token_counts.items(), key=lambda x: -x[1]):
    vocab.append(token)
  if VOCAB_SIZE is not None:
    vocab = vocab[:VOCAB_SIZE]
  with open('vocab.txt', 'w') as f:
    f.write('\n'.join(vocab))
tok_to_id = {t:i for i, t in enumerate(vocab)}
assert len(tok_to_id) == len(vocab)

print('Tokenizing')
for data_split, data in zip(['train', 'valid', 'test'], [train_data, valid_data, valid_data]):
  max_length = 0
  lines = []
  for doc, _ in tqdm(data):
    title, doc = doc.splitlines()
    doc_tokens = word_tokenize(doc.strip().lower())
    doc_tokens = [('<UNK>' if t not in tok_to_id else t) for t in doc_tokens]
    if len(doc_tokens) > max_length:
      max_length = len(doc_tokens)
    lines.append(' '.join(doc_tokens))

  with open('roc.{}.txt'.format(data_split), 'w') as f:
    f.write('\n'.join(lines))
  print(max_length)
