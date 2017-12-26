from nltk import RegexpTokenizer
from nltk.corpus import stopwords


class Clear(object):
    def __init__(self, docs_data):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stop_word_set = set(stopwords.words('english'))
        self.docs_data = docs_data

    def clean(self):
        new_doc = []
        for doc_data in self.docs_data:
            doc_data = doc_data.lower()
            doc_data = self.tokenizer.tokenize(doc_data)
            doc_data = list(set(doc_data).difference(self.stop_word_set))
            new_doc.append(doc_data)
        return new_doc
