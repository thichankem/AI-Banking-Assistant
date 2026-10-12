import math
import re

class bm25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1=k1
        self.b=b
        self.corpus=corpus
        self.doc_count=len(corpus)

        self.tokenized_corpus = []
        self.doc_len = []

        for doc in corpus:
            text=doc["content"]
            tokens=re.findall(r'\w+', text.lower())
            doc_len=len(tokens)

            self.tokenized_corpus.append(tokens)
            self.doc_len.append(doc_len)

        self.avgdl=sum(self.doc_len)/self.doc_count if self.doc_count>0 else 0

        self.idf=self.compute_idf()

    
    def compute_idf(self):
        word_doc_freq={}
        for tokens in self.tokenized_corpus:
            for word in set(tokens):
                word_doc_freq[word]=word_doc_freq.get(word, 0)+1

        idf={}

        for word, freq in word_doc_freq.items():
            val=math.log(((self.doc_count-freq+0.5)/(freq+0.5))+1)
            idf[word]=val
        
        return idf


    def get_scores(self, query):
        query_tokens=re.findall(r'\w+', query.lower())
        scores=[0.0]*self.doc_count

        for i, doc_tokens in enumerate(self.tokenized_corpus):
            doc_len=self.doc_len[i]
            if doc_len==0:
                continue

            tf={}
            for token in doc_tokens:
                tf[token]=tf.get(token, 0)+1

            doc_score=0

            for q_term in query_tokens:
                if q_term not in tf:
                    continue
                
                freq=tf[q_term]
                idf=self.idf.get(q_term, 0)

                numerator=freq*(self.k1+1)
                denominator=freq+self.k1*(1-self.b+self.b*(doc_len/self.avgdl))

                doc_score+=idf*(numerator/denominator)

            scores[i]=doc_score

        return scores


    def bm25search(self, query, top_k=5):
        scores=self.get_scores(query)

        results=[]
        for i, score in enumerate(scores):
            if score>0:
                item=dict(self.corpus[i])
                item["bm25_score"]=round(score, 4)

                results.append(item)

        results.sort(key=lambda x:x["bm25_score"], reverse=True)
        return results[0:top_k]
