import jsonlines
from sentence_transformers import SentenceTransformer
import pickle
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--json-path', type=str, help="path to json file")
parser.add_argument('--pkl-path', type=str, help="path to pkl file (to save)")
args = parser.parse_args()

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = []
cnt = 0
with jsonlines.open(args.json_path) as reader:
    for obj in reader:
        cnt += 1
        if cnt % 1000000 == 0:
            print(f"{cnt}")
        text = obj.get('reviewText', None)
        if text is not None:
            sentences.append(text)
embeddings = model.encode(sentences, show_progress_bar=True)
print(f"data shape: {embeddings.shape}")
 
with open(args.pkl_path, 'wb') as f:
    pickle.dump(embeddings, f)
