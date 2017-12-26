import gensim
from os import listdir
import Docs

# casesLabels = [f for f in listdir('./dataset_all/') if f.endswith('.txt')]
casesLabels = [f for f in listdir('./dataset/') if f.endswith('.txt')]

# print casesLabels

data = []
for doc in casesLabels:
    # data.append(open('./dataset_all/' + doc).read())
    data.append(open('./dataset/' + doc).read())

cleaner = Docs.Clear(data)
data = cleaner.clean()

#print data

docs_iterator = Docs.Labeling(data, casesLabels)
# print docs_iterator
#
# for d in docs_iterator:
#     print d

model = gensim.models.Doc2Vec(alpha=.025, min_alpha=.025, min_count=1)
model.build_vocab(docs_iterator)

for epoch in range(100):
    model.train(docs_iterator, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.002
    model.min_alpha = model.alpha


model.save('selected_cases.model')
