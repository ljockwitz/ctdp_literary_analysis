# Current Topics in Digital Philology - Literary Analysis 🔏

## 👋 Description
This repository is part of the `Current Topics in Digital Philology` course in the Language Technology programme
at Uppsala University in autumn term 2023.

## Installation

After downloading the repository, you can install the project like this:

    pip install --editable .

## 📚 Requirements

As the python code in this repository is written in python 3.9,
this or a more recent version is highly recommended. Compatibility with older
versions of python can't be guaranteed. Additionally, the code requires the
following packages:

- nltk
- gensim
- pyldavis
- spacy
- matplotlib
- numpy

You can install the requirements from `requirements.txt`:

    pip install -r requirements.txt

Additionally, you need to install the German language model `de_core_news_sm`:

    python -m spacy download de_core_news_sm

## Data

The data is retrieved from [Project Gutenberg](https://www.gutenberg.org). Heading and license 
were manually removed from each `txt`-file but the latter can still be found in `/data/LICENSE.txt`.

This project includes the following works:

- [_Die Leiden des jungen Werther - Band 1_](https://www.gutenberg.org/ebooks/2407), Johann Wolfgang von Goethe
- [_Die Leiden des jungen Werther - Band 2_](https://www.gutenberg.org/ebooks/2408), Johann Wolfgang von Goethe
- [_Die Räuber_](https://www.gutenberg.org/ebooks/47804), Friedrich Schiller
- [_Kabale und Liebe_](https://www.gutenberg.org/ebooks/6498), Friedrich Schiller
- [_Götz von Berlichingen_](https://www.gutenberg.org/ebooks/2321), Johann Wolfgang von Goethe
- [_Iphigenie auf Tauris_](https://www.gutenberg.org/ebooks/2054), Johann Wolfgang von Goethe
- [_Novelle_](https://www.gutenberg.org/ebooks/2320), Johann Wolfgang von Goethe
- [_Reineke Fuchs_](https://www.gutenberg.org/ebooks/2228), Johann Wolfgang von Goethe

## 🖇️ Tools

`lab3.py` contains the code for Lab 3 and offers classes and functions to preprocess the data, perform topic modeling 
and visualize the results.

`final.py` contains the code for the final course report and provides functions to 
compute POS-tag distributions and average word and sentence length as well as to plot the results.

## 📝 Authors and acknowledgment
This repository is maintained by: 

- [Lisa Jockwitz](mailto:lisa.jockwitz.8807@student.uu.se)


## ⏳ Project status

Submitted on 31st December 2023.