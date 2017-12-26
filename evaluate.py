import gensim

# cases_model = gensim.models.doc2vec.Doc2Vec.load('cases.model')
cases_model = gensim.models.doc2vec.Doc2Vec.load('selected_cases.model')

# docvec = cases_model.docvecs[1]
# print docvec

# similar_doc = cases_model.docvecs.most_similar(14)
# print similar_doc

sims = cases_model.docvecs.most_similar('abdan-v-deen-lawnet.txt')
print sims

# case = cases_model.docvecs.most_similar(["homuside"])
# print case