import nltk.corpus as corpus
from nltk.stem import *
import sys
parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
    from cprTfIdf import CorpusReader_TFIDF
else:
    from .cprTfIdf import CorpusReader_TFIDF

# Note: the code uses scipy, string, math libraries


corp1 = CorpusReader_TFIDF(corpus.brown,PorterStemmer(),'raw','base',"Yes","no")
corp1.displayResults()

corp2 = CorpusReader_TFIDF(corpus.shakespeare,PorterStemmer(),'raw','base',"Yes","no")
corp2.displayResults()

corp3 = CorpusReader_TFIDF(corpus.state_union,PorterStemmer(),'raw','base',"Yes","no")
corp3.displayResults()