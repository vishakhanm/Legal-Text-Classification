import numpy as np
import pandas as pd
import pickle
import sklearn.utils
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_json('combined.json', lines=True)

tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 8000,
    stop_words = 'english'
)
tfidf.fit(data.contents)
text = tfidf.transform(data.contents)

clusters = MiniBatchKMeans(n_clusters=14, init_size=1024, batch_size=2048, random_state=20).fit_predict(text)

actual_list = []
cluster_list = []
labels = tfidf.get_feature_names()
n_terms = 10
# def get_top_keywords(data, clusters, labels, n_terms):
df = pd.DataFrame(text.todense()).groupby(clusters).mean()

for i,r in df.iterrows():
    # print('\nCluster {}'.format(i))
    cluster_list.append(f"Cluster {i}")
    # print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
    actual_list.append([labels[t] for t in np.argsort(r)[-n_terms:]])

# get_top_keywords(text, clusters, tfidf.get_feature_names(), 10)
# print(actual_list)

pickle.dump(actual_list, open('model.pkl','wb'))

