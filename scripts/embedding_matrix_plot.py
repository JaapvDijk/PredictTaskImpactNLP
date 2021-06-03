import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

# plt.clf()

embedding_model = Word2Vec.load("../embedding_models/google+labela.model")
texts = np.array(['javascript', 'python', 'ontwikkelaar', 'developer', 'message', 'notification', 'authentication', 'oauth'])
embeddings = []
for word in texts:
    embeddings.append(embedding_model[word])

pca = PCA(n_components=2)
embeddings_reduced = pca.fit_transform(embeddings)

xs = [i[0] for i in embeddings_reduced]
ys = [i[1] for i in embeddings_reduced]

plt.plot(xs,ys,'bo')

for x,y,text in zip([i[0] for i in embeddings_reduced], [i[1] for i in embeddings_reduced], texts):
    print(text)
    plt.annotate(text,
                 (x,y),
                 textcoords="offset points",
                 xytext=(0,8),
                 ha='center')
plt.show()
