import re
import spacy
import matplotlib.pyplot as plt
import numpy as np

from lab3 import Corpus, prepare_for_lda, train_lda, visualize
from collections import Counter

nlp = spacy.load("de_core_news_sm")


def pos_dist(spacy_doc):
    """Compute pos tag distribution based on the universal tag set

    :param spacy_doc: a document processed by a spacy pipeline
    :return: dictionary
    """
    all_pos = []
    for token in spacy_doc:
        all_pos.append(token.pos_)
    total_pos = len(all_pos)

    # Calculate the frequency of each pos tag
    pos_counts = Counter(all_pos)

    # Calculate the relative frequency of each term
    relative_frequencies_pos = dict()
    for term, count in pos_counts.items():
        relative_frequencies_pos[term] = round(count / total_pos, 3) * 100

    return relative_frequencies_pos


def plot_pos_dist(filename, pos_tags, pos_dist_dict):
    x = np.arange(len(pos_tags))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    fig.set_figwidth(10)

    for tag, measurement in pos_dist_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=tag)
        ax.bar_label(rects, padding=3, fontsize=6)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage')
    ax.set_title('POS-tag distribution')
    ax.set_xticks(x + width, pos_tags)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 20)
    plt.savefig("./results/" + filename)


def compute_statistics(spacy_doc):
    """Compute average word length, average sentence length per word and average sentence length by character.

    :param spacy_doc: a document processed by a spacy pipeline
    :return: tuple of the computed scores
    """
    token_count = len(list(spacy_doc))
    letter_count = 0
    sentence_count = len(list(spacy_doc.sents))
    for token in spacy_doc:
        letter_count += len(token)

    avg_word_lgth = round(letter_count / token_count, 2)
    avg_sent_lgth_word = round(token_count / sentence_count, 2)
    avg_sent_lgth_letter = round(letter_count / sentence_count, 2)

    return avg_word_lgth, avg_sent_lgth_word, avg_sent_lgth_letter


if __name__ == "__main__":
    # number of topics k
    corpus_wk = Corpus()
    books_wk = [("goethe_iphigenie_auf_tauris.txt", "iph", True),
                ("goethe_novelle.txt", "nov", False),
                ("goethe_reineke_fuchs.txt", "rei", False)
                ]
    for file, key, is_play in books_wk:
        corpus_wk.add_book("./data/" + file, key, is_play)
    corpus_wk.remove_speakers()

    corpus_sud = Corpus()
    books_sud = [("goethe_die_leiden_des_jungen_werther_1.txt", "wer1", False),
                 ("goethe_die_leiden_des_jungen_werther_2.txt", "wer2", True),
                 ("goethe_goetz_von_berlichingen.txt", "goe", True)]
    for file, key, is_play in books_sud:
        corpus_sud.add_book("./data/" + file, key, is_play)
    corpus_sud.remove_speakers()

    id2word_wk, lda_corpus_wk, tok_corpus_wk = prepare_for_lda(corpus_wk.to_list())

    for k in range(5, 7):
        lda_model_wk, doc_lda_wk = train_lda(id2word_wk, lda_corpus_wk, k)
        for topic, word_list in lda_model_wk.print_topics(num_words=20):
            word_list = re.findall(r"(?:\*\")(.*?)(?:\")", word_list)
            print(topic, " & ", " ".join(word_list), " \\\\")

        print(" ")

        visualize("wk", k, lda_model_wk, lda_corpus_wk, id2word_wk)

    # stylistic analysis

    pos_dist_dict = {}
    stats_dict = {}
    for key, book in corpus_wk.books.items():
        doc = nlp(book)
        pos_dist_dict[key] = pos_dist(doc)
        stats_dict[key] = compute_statistics(doc)

    for key, book in corpus_sud.books.items():
        doc = nlp(book)
        pos_dist_dict[key] = pos_dist(doc)
        stats_dict[key] = compute_statistics(doc)

    categories = ("Avg word lgth", "Avg sent lgth (word)", "Avg sent lgth (letter)")

    x = np.arange(len(categories))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for title, measurement in stats_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=title)
        ax.bar_label(rects, padding=3, fontsize=6)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg')
    ax.set_title('Text statistics')
    ax.set_xticks(x + width, categories)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 140)
    plt.savefig("./results/text_stats.png")

    # plot pos tag distribution

    pos_tags_1 = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM")
    pos_tags_2 = ("PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X")

    pos_dist_dict_1 = {}
    pos_dist_dict_2 = {}
    for key, dist in pos_dist_dict.items():
        values = []
        for tag in pos_tags_1:
            if tag in dist.keys():
                values.append(dist[tag])
            else:
                values.append(0)
        pos_dist_dict_1[key] = values
        values = []
        for tag in pos_tags_2:
            if tag in dist.keys():
                values.append(dist[tag])
            else:
                values.append(0)
        pos_dist_dict_2[key] = values

    plot_pos_dist("pos_dist_1.png", pos_tags_1, pos_dist_dict_1)
    plot_pos_dist("pos_dist_2.png", pos_tags_2, pos_dist_dict_2)
