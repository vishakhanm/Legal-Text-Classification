import os
import numpy as np
import pandas as pd
import pickle
import law_dict as l
import sklearn.utils
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from uuid import uuid4
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)
actual_list = pickle.load(open('model.pkl', 'rb'))

def get_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    new_list = []
    for i,r in df.iterrows():
        # print('\nCluster {}'.format(i))
        # print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
        new_list.append([labels[t] for t in np.argsort(r)[-n_terms:]])
    return new_list


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    output = []
    newd = request.form['text']
    new_data = pd.DataFrame({"contents": newd }, index =[0])

    new_tfidf = TfidfVectorizer(
    max_features = 10,
    stop_words = 'english'
    )
    new_tfidf.fit(new_data.contents)
    new_text = new_tfidf.transform(new_data.contents)
    new_clusters = MiniBatchKMeans(n_clusters=1, init_size=1024, batch_size=2048, random_state=20).fit_predict(new_text)

    new_list=get_keywords(new_text, new_clusters, new_tfidf.get_feature_names(), 10)
    

    for i in range(len(actual_list)):
        count = 0
        for k in range(len(new_list[0])):
            if new_list[0][k] in actual_list[i]:
                count += 1
            # print(new_list[0][k])
        
        if count >=2:
            # print(list(l.law_dict.keys())[list(l.law_dict.values()).index(actual_list[i])])
            output.append(list(l.law_dict.keys())[list(l.law_dict.values()).index(actual_list[i])])
        
    if(len(output) == 0):
        return render_template('index.html', prediction_text= 'N.A. ', data = newd)
    else:
        return render_template('index.html', prediction_text= ', '.join(output), data = newd)




if __name__ == "__main__":
    app.run(debug=False)



