import gensim
import nltk
import spacy
import string
import re

from pprint import pprint
from gensim import corpora
from nltk.corpus import stopwords
# nltk.download("stopwords")

nlp = spacy.load("de_core_news_sm")


class Corpus:
    def __init__(self):
        self.books = {}
        self.count = 0
        self.is_play = {}
        self.sents = {}
        self.words = {}

    def add_book(self, path: str, key: str, is_play=False):
        """Read file and perform some simple preprocessing

        :param path: path to txt-file
        :param key: short key to refer to the book
        :param is_play: define if book is a play
        :return: None
        """
        with open(path, "r", encoding="utf-8") as f:
            book = f.read()

        if is_play:
            # mark speakers with ** : if book is a play so that it can be removed easily later on
            # move stage directions behind
            # book = re.sub(r"(Personen:\n\n.*?\n\n)", r"\1\.", book, re.DOTALL)
            book = re.sub(r"(\n)((?:[A-ZÖÄÜ]\w+ ?)+)( \(.*?\))?\.", r"\1**\2:\3", book)

        # remove new lines
        book = re.sub(r"\s+", r" ", book)
        self.books[key] = book
        self.is_play[key] = is_play
        self.count += 1

    def remove_speakers(self):
        for key, text in self.books.items():
            if self.is_play[key]:
                text = re.sub(r"\*\*([A-ZÖÄÜ]\w+\s?)+:", "", text)
                self.books[key] = text

    def tokenize(self):
        for key, text in self.books.items():
            self.sents[key] = nltk.sent_tokenize(text, "german")
            self.words[key] = nltk.word_tokenize(text, "german")

    def to_list(self):
        return list(self.books.values())


def prepare_for_lda(data: list):

    puncts = list(string.punctuation)
    german_stop_words = stopwords.words('german')
    german_stop_words.extend(["--", "sollen", "sagen", "gehen", "wohl", "kommen", "sehen", "tun",
                              "lassen", "ab", "wer"])

    proc_data = []
    for book in data:
        doc = nlp(book)
        proc_data.append(doc)

    # to decide: lower case?
    proc_data = [[w.lemma_ for w in book
                  if w.text not in puncts and w.text != " " and w.lemma_ not in german_stop_words]
                 for book in proc_data]

    # Create Dictionary
    id2word = corpora.Dictionary(proc_data)
    # Create Corpus
    texts = proc_data
    # Term Document Frequency
    lda_corpus = [id2word.doc2bow(text) for text in texts]

    return id2word, lda_corpus


def train_lda(id2word, corpus, num_topics=10):
    # Build LDA model
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics,
                                       alpha=50 / num_topics,
                                       eta=0.1,
                                       chunksize=1,
                                       minimum_probability=0.001,
                                       passes=5
                                       )
    doc_lda = lda_model[corpus]

    return lda_model, doc_lda


if __name__ == "__main__":
    corpus_1 = Corpus()
    corpus_1.add_book("./data/goethe_goetz_von_berlichingen.txt", "goetz", True)
    corpus_1.add_book("./data/goethe_die_leiden_des_jungen_werther.txt", "werther")
    corpus_1.remove_speakers()

    id2word, lda_corpus = prepare_for_lda(corpus_1.to_list())
    # print(prepared[0][200:300])
    lda_model, doc_lda = train_lda(id2word, lda_corpus, 5)
    pprint(lda_model.print_topics(num_words=20))
