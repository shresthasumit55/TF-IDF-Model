import nltk.corpus as corp
from nltk.probability import FreqDist
import math
import string
from scipy import spatial
from nltk.stem import *


class CorpusReader_TFIDF:

    def __init__(self,corp,stemmer,tf='raw',idf='base',stopword="Yes",ignorecase="Yes"):
        self.corpus=corp
        self.words=self.collectWords()
        self.files = self.corpus.fileids()
        self.stemmer=stemmer
        self.stopword = stopword
        self.ignorecase=ignorecase
        self.preProcess(stemmer,stopword,ignorecase)
        self.TFMode,self.IDFMode = self.setTFIDFMode(tf,idf)
        self.numberOfDocuments=len(self.corpus.fileids())
        self.wordCountPerDocument = self.termFrequency()
        self.inverseCount = self.inverseCount()
        self.tfIdf = self._calculateTFIDFvalues()


    # This function returns all the words in the corpus
    def collectWords(self):
        templist=[]
        for file in self.corpus.fileids():
            for item in self.corpus.words(file):
                templist.append(item)
        return templist


    # This method performs preprocessing on the words of the corpus based on the passed parameters
    # Input: Type of stemmer, stopword configuration[Yes/No] and ignorecase configuration(Yes,No)
    # Output: It modifies the words of the corpus based on the preprocessing configuration
    def preProcess(self,stemmer,stopword,ignorecase):
        if stopword!="none":
            self.removeStopWords()
        self.stem()
        if ignorecase!="no":
            self.words = [word.lower() for word in self.words]

    # This method sets the tf/idf mode
    # Input TF mode: raw,log or binary and idf mode
    # Output Integer values for those modes
    def setTFIDFMode(self,tf,idf):
        tfMode={'raw':1,'log':2,'binary':3}
        idfMode={'base':1,'smooth':2,'prob':3}
        return tfMode.get(tf,1),idfMode.get(idf,1)

    # This method removes stopwords from the corpus
    # The stop words are from the nltk stopwords corpus which is placed in a file
    def removeStopWords(self):
        stopwords=list(set(corp.stopwords.words('english'))) + list(string.punctuation)
        filteredwords = [word for word in self.words if not word in stopwords]
        self.words = filteredwords

    # This method stems each word of the corpus
    def stem(self):
        self.words= ([self.stemmer.stem(word) for word in self.words])

    # This method returns count of terms in each document
    # Input is the entire corpus
    # Returns the count of terms in each document
    def termFrequency(self):
        uniqueWords = set(self.words)
        freq= {}
        count={}
        for fileid in self.corpus.fileids():
            fileWords = self.corpus.words(fileid)
            fileWords = self.preProcessWordList(fileWords)
            freq[fileid] = dict(FreqDist([self.stemmer.stem(word) for word in fileWords]))
            count[fileid]={}
        while len(uniqueWords)>0:
            term = uniqueWords.pop()
            for fileid in self.corpus.fileids():
                count[fileid][term]=[0 if(freq[fileid].get(term)==None) else freq[fileid][term],
                                     0 if(freq[fileid].get(term)==None) else 1+math.log2(freq[fileid][term]),
                                     0 if(freq[fileid].get(term)==None) else 1]
        return count

    # This method returns number of documents which contains the words of the corpus ( inverse count)
    # Input is the entire corpus
    # Returns number of documents which contain the word for each word in corpus
    def inverseCount(self):
        uniqueWords=set(self.words)
        inverseCount = {}
        while len(uniqueWords)>0:
            term = uniqueWords.pop()
            tempCount=0
            for fileId in self.corpus.fileids():
                if self.wordCountPerDocument[fileId][term][0]>0:
                    tempCount=tempCount + 1
            if float((self.numberOfDocuments-tempCount)/tempCount)<0:
                print("whatwhat")
            #inverseCount holds the three values of idf: base, smoothed and probabilistic inverse for each term
            inverseCount[term] = [math.log2(float(self.numberOfDocuments/tempCount)),
                                  math.log2(float(1+float(self.numberOfDocuments/tempCount))),
                                  0 if (self.numberOfDocuments-tempCount==0) else math.log2(float((self.numberOfDocuments-tempCount)/tempCount))]
        return inverseCount

    # Return a list of ALL tf-idf vector for the corpus, ordered by the order where filelds are returned
    # If fileid parameter is passed then the function returns tf-idf vector corresponding to that file
    # If list of filed id is passed then it returns a list of vectors, corresponding to the tf-idf to the list of
    # fileid input
    def tf_idf(self,fileid=None):
        if fileid is not None:
            if type(fileid)==str:
                return self._getTFIDFofFile(fileid)
            else:
                result={}
                for id in fileid:
                    result[id]=self._getTFIDFofFile(id)
                return result
        else:
            resultList=[self.tfIdf.get(file) for file in self.files]
            return resultList

    # This is the method in which actual tf-idf values are calculated
    # Input is the entire corpus
    # Output is the tf-idf values between all the documents and words
    def _calculateTFIDFvalues(self):
        words = set(self.words)
        tfIdfvalues = {}
        for fileId in self.corpus.fileids():
            words=set(self.words)
            tempVector=[]
            while len(words)>0:
                term = words.pop()
                tempVector.append(float(self.wordCountPerDocument[fileId].get(term, 0)[self.TFMode] * self.inverseCount[term][self.IDFMode]))
            tfIdfvalues[fileId]=tempVector
        return tfIdfvalues

    # the function returns tf-idf vector corresponding to a file
    # Input fileid of the corpus
    # Output tf idf vector of that file
    def _getTFIDFofFile(self,fileId):
        return self.tfIdf.get(fileId)

    # return the list of the words in the order of the dimension of each corresponding to each vector of
    # the tf-idf vector
    # Input is the entire corpus
    # Output: list of words in the corpus
    def tf_idf_dim(self):
        return set(self.words)

    # return the cosine similarity between two documents in the corpus
    # Input: two files in the corpus
    # Output: cosine similarity between those files
    def cosine_sim(self,fileids):

        vectors=[self.tfIdf.get(file) for file in fileids]
        similarity = 1-spatial.distance.cosine(vectors[0],vectors[1])
        return similarity

    # this function performs stopword removal and change cases to lowercase if ignorecase is set to yes
    # Input: a list of words (query)
    # Output: preprocessed list with stopwords removal and case of the words changed based on the stopword and
    # ignorecase value passed to the constructor
    def preProcessWordList(self,wordlist):
        if self.stopword!="none":
            stopwords = list(set(corp.stopwords.words('english'))) + list(string.punctuation)
            filteredwords = [word for word in wordlist if not word in stopwords]
            wordlist = filteredwords
        if self.ignorecase!="no":
            wordlist = [word.lower() for word in wordlist]
        return wordlist

    # returns a tf-idf vector of a query
    # Input: list of words
    # Output: tf-idf vector of that list of words which is treated as a document
    def tf_idf_new(self,wordList):
        wordList = self.preProcessWordList(wordList)
        wordCount = FreqDist([self.stemmer.stem(word) for word in wordList])
        words=set(self.words)
        tf_idf=[]
        termFrequency={}
        for word in words:
            termFrequency[word]=[wordCount.get(word,0),1+math.log2(wordCount.get(word,0.5)),0 if wordCount.get(word,0)==0 else 1]
            tf_idf.append(((self.inverseCount[word][self.IDFMode])+1)*termFrequency[word][self.TFMode])
        return tf_idf

    # returns cosine similarity between a query and file
    # Input: list of words, file
    # Output: cosine similarity between a query and file
    def cosine_sim_new(self,wordlist,fileid):
        wordlist = self.preProcessWordList(wordlist)
        wordlist = [self.stemmer.stem(word) for word in wordlist]
        queryTFIDF = self.tf_idf_new(wordlist)
        documentTFIDF=self._getTFIDFofFile(fileid)
        for word in wordlist:
            if word in self.words:
                index=list(self.tf_idf_dim()).index(word)
                documentTFIDF[index]=((self.inverseCount[word][self.IDFMode])+1)*self.wordCountPerDocument[fileid][word][self.TFMode]
        return 1-spatial.distance.cosine(queryTFIDF,documentTFIDF)

    # This method is a wrapper method for displaying the output
    # Input: the entire corpus
    # Output: the results in proper format
    def displayResults(self):
        print(self.corpus)
        print(list(self.tf_idf_dim())[:15])
        for file in self.files:
            print(file, ',',*self.tf_idf(file)[:15])
        for file in self.files:
            for file2 in self.files:
                print(file,file2,self.cosine_sim([file,file2]))