import gensim
import nltk
import spacy
import string
import re
import pyLDAvis.gensim
import pickle
import pyLDAvis
import os

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
            book = re.sub(r"[~_]", r"", book)
            book = re.sub(r"--", r"-", book)
            book = re.sub(r"=\w.*?\w=", r"", book)
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

    tok_data = []
    for book in data:
        doc = nlp(book)
        tok_data.append(doc)

    proc_data = [[w.lemma_ for w in book
                  if w.text not in puncts and w.text != " " and w.lemma_ not in german_stop_words
                  and w.pos_ in ["VERB", "NOUN", "ADJ"]]
                 for book in tok_data]

    # chunking
    chunked_data = []
    for book in proc_data:
        n = 20
        chunks = [book[i:i + n] for i in range(0, len(book), 3)]
        chunked_data.extend(chunks)

    # Create Dictionary
    id2word = corpora.Dictionary(chunked_data)
    # Create Corpus
    texts = chunked_data
    # Term Document Frequency
    lda_corpus = [id2word.doc2bow(text) for text in texts]

    return id2word, lda_corpus, tok_data


def train_lda(id2word, corpus, num_topics=10):
    # Build LDA model
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics,
                                       alpha=0.01,
                                       eta=0.9,
                                       chunksize=100,
                                       passes=4
                                       )
    doc_lda = lda_model[corpus]

    return lda_model, doc_lda


def visualize(filename, k, lda_model, lda_corpus, id2word):
    # Visualize the topics
    # pyLDAvis.enable_notebook()
    ldavis_data_filepath = os.path.join('./results/' + filename + "_" + str(k))

    ldavis_prepared = pyLDAvis.gensim.prepare(lda_model, lda_corpus, id2word)
    with open(ldavis_data_filepath, 'wb') as f:
        pickle.dump(ldavis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(ldavis_data_filepath, 'rb') as f:
        ldavis_prepared = pickle.load(f)
    pyLDAvis.save_html(ldavis_prepared, './results/' + filename + str(k) + '.html')


if __name__ == "__main__":
    # number of topics k
    corpus_all = Corpus()
    books = [("goethe_die_leiden_des_jungen_werther_1.txt", "wer1", False),
             ("goethe_die_leiden_des_jungen_werther_2.txt", "wer2", False),
             ("goethe_goetz_von_berlichingen.txt", "goe", True),
             ("schiller_die_raeuber.txt", "rae", True),
             ("schiller_kabale_und_liebe.txt", "kab", True)
             ]
    for file, key, is_play in books:
        corpus_all.add_book("./data/" + file, key, is_play)
    corpus_all.remove_speakers()

    id2word, lda_corpus, _ = prepare_for_lda(corpus_all.to_list())

    for k in range(3, 7):
        # print(prepared[0][200:300])
        lda_model, doc_lda = train_lda(id2word, lda_corpus, k)
        for topic, word_list in lda_model.print_topics(num_words=20):
            word_list = re.findall(r"(?:\*\")(.*?)(?:\")", word_list)
            print(topic, " & ", " ".join(word_list), " \\\\")
        print(" ")

        visualize("all", k, lda_model, lda_corpus, id2word)

    print(f"\n\nDie Leiden des jungen Werther\n\n")

    id2word_wer, lda_corpus_wer, _ = prepare_for_lda([corpus_all.books["wer1"], corpus_all.books["wer2"]])
    for k in range(3, 7):
        lda_model_wer, doc_lda_wer = train_lda(id2word_wer, lda_corpus_wer, k)
        for topic, word_list in lda_model_wer.print_topics(num_words=20):
            word_list = re.findall(r"(?:\*\")(.*?)(?:\")", word_list)
            print(topic, " & ", " ".join(word_list), " \\\\")
        print(" ")
