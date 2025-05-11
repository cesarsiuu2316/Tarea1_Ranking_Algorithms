import numpy as np

def load_dataset(path):
    X = [] # Features
    y = [] # relevance labels
    qids = [] # Query IDs
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split() # Delete leading and trailing whitespace and split by whitespace
            y.append(int(parts[0])) # Parse relevance label
            qid = int(parts[1].split(':')[1]) # Parse query ID and get rid of the 'qid:' prefix
            qids.append(qid)
            features = []
            for x in parts[2:]:
                feature = float(x.split(':')[1]) # Parse each feature and get only the value
                features.append(feature) # Append the feature value to the list
            X.append(features) 
    return np.array(X), np.array(y), np.array(qids)