### Code copied from: https://github.com/centre-for-humanities-computing/fabula_pipeline/blob/main/src/utils.py
import re
import pandas as pd
import numpy as np
from collections import Counter
import spacy
import bz2
import gzip
from nltk.tokenize import word_tokenize
from afinn import Afinn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import rpy2.robjects.packages as rpackages
# from rpy2.robjects import conversion, default_converter
import src.saffine.multi_detrending as md
import src.roget.roget as roget
import textstat
from math import log

def get_nlp(lang: str):
    """
    checks if the spacy model is loaded, errors if not
    """
    if lang == "english":
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise OSError(
                "en_core_web_sm not downloaded, run python3 -m spacy download en_core_web_sm"
            ) from e

    elif lang == "danish":
        try:
            nlp = spacy.load("da_core_news_sm")

        except OSError as e:
            raise OSError(
                "da_core_news_sm not downloaded, run python3 -m spacy download da_core_news_sm"
            ) from e

    return nlp


def get_spacy_attributes(token):
    # Save all token attributes in a list
    token_attributes = [
        token.i,
        token.text,
        token.lemma_,
        token.is_punct,
        token.is_stop,
        token.morph,
        token.pos_,
        token.tag_,
        token.dep_,
        token.head,
        token.head.i,
        token.ent_type_,
    ]

    return token_attributes


def create_spacy_df(doc_attributes: list) -> pd.DataFrame:
    df_attributes = pd.DataFrame(
        doc_attributes,
        columns=[
            "token_i",
            "token_text",
            "token_lemma_",
            "token_is_punct",
            "token_is_stop",
            "token_morph",
            "token_pos_",
            "token_tag_",
            "token_dep_",
            "token_head",
            "token_head_i",
            "token_ent_type_",
        ],
    )
    return df_attributes


def filter_spacy_df(df: pd.DataFrame) -> pd.DataFrame:
    spacy_pos = ["NOUN", "VERB", "ADJ", "INTJ"]

    filtered_df = df.loc[
        (df["token_is_punct"] == False)
        & (df["token_is_stop"] == False)
        & (df["token_pos_"].isin(spacy_pos))
    ]

    filtered_df = filtered_df.copy()

    filtered_df["token_roget_pos_"] = filtered_df["token_pos_"].map(
        {"NOUN": "N", "VERB": "V", "ADJ": "ADJ", "INTJ": "INT"}
    )
    return filtered_df


def get_token_categories(df: pd.DataFrame) -> str:
    token_categories = df.apply(
        lambda row: roget.categories(str(row["token_lemma_"]), row["token_roget_pos_"]),
        axis=1,
    ).to_string()

    return token_categories



def avg_wordlen(words: list[str]) -> float:
    """
    calculates average wordlength from a list of words
    """
    len_all_words = [len(word) for word in words]
    avg_word_length = sum(len_all_words) / len(words)
    return avg_word_length


def avg_sentlen(sents: list[str]) -> float:
    """
    calculates average sentence length from a list of sentences
    """
    avg_sentlen = sum([len(sent) for sent in sents]) / len(sents)
    return avg_sentlen


def compressrat(sents: list[str]):
    """
    Calculates the GZIP compress ratio and BZIP compress ratio for the first 1500 sentences in a list of sentences
    """
    # skipping the first that are often title etc
    selection = sents[2:1502]
    asstring = " ".join(selection)  # making it a long string
    encoded = asstring.encode()  # encoding for the compression

    # GZIP
    g_compr = gzip.compress(encoded, compresslevel=9)
    gzipr = len(encoded) / len(g_compr)

    # BZIP
    b_compr = bz2.compress(encoded, compresslevel=9)
    bzipr = len(encoded) / len(b_compr)

    return gzipr, bzipr


def cleaner(text: str, lower=False) -> str:
    
    text = re.sub(r"[0-9]", "", text)
    text = re.sub(r'[,.;:"?!*()\']', "", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub(r"[^a-zA-Z ]+", "", text)

    if lower:
        text = text.lower()
    return text


def text_entropy(text: str, language: str, base=2, asprob=True, clean=True):
    
    if clean:
        text = cleaner(text)

    words = word_tokenize(text, language=language)
    total_len = len(words) - 1
    bigram_transform_prob = Counter()
    word_transform_prob = Counter()

    # loop through each word in the cleaned text and calculate the probability of each bigram
    for i, word in enumerate(words):
        if i == 0:
            word_transform_prob[word] += 1

            # very first word gets assigned as first pre
            pre = word
            continue

        word_transform_prob[word] += 1
        bigram_transform_prob[(pre, word)] += 1
        pre = word

    # return transformation probability if asprob is set to true
    if asprob:
        return word_transform_prob, bigram_transform_prob
    
    # if not, calculate the entropy and return that
    if not asprob:
        log_n = log(total_len, base)

        bigram_entropy = cal_entropy(base, log_n, bigram_transform_prob)
        word_entropy = cal_entropy(base, log_n, word_transform_prob)

        return bigram_entropy / total_len, word_entropy / total_len

def cal_entropy(base, log_n, transform_prob):
    entropy = sum([-x * (log(x, base) - log_n) for x in transform_prob.values()])
    return entropy

# def prepare_syuzhet():
#     # import R's utility package
#     with conversion.localconverter(default_converter):
#         utils = rpackages.importr("utils")

#         # select a mirror for R packages
#         utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

#         # install the package if is not already installed
#         if not rpackages.isinstalled("syuzhet"):
#             utils.install_packages("syuzhet")

#         return None

def get_sentarc(sents: list[str], sent_method: str, lang: str) -> list[float]:
    """
    Create a sentiment arc from a list of sentences.
    Sent_method can be either vader, syuzhet, or avg_syuzhet_vader.
    Lang can be either english or danish.
    """
    if "afinn" in sent_method:
        afinn = Afinn(language=lang[:2])
        afinn_arc = [afinn.score(sentence) for sentence in sents]

        return afinn_arc

    if "vader" in sent_method:
        sid = SentimentIntensityAnalyzer()

        vader_arc = []
        for sentence in sents:
            compound_pol = sid.polarity_scores(sentence)["compound"]
            vader_arc.append(compound_pol)

        if "avg" not in sent_method:
            return vader_arc

    # if "syuzhet" in sent_method:
    #     prepare_syuzhet()
    #     with conversion.localconverter(default_converter):
    #         syuzhet = rpackages.importr("syuzhet")
    #         syuzhet_arc = list(syuzhet.get_sentiment(sents, method="syuzhet"))

    #         if "avg" not in sent_method:
    #             return syuzhet_arc

    # if "avg" in sent_method:
    #     sent_array = np.array([vader_arc, syuzhet_arc])
    #     arc = list(np.mean(sent_array, axis=0))

    #     return arc


def divide_segments(arc: list[float], n: int):
    """
    divide a list of floats into segments of the specified number of items
    """
    for i in range(0, len(arc), n):
        yield arc[i : i + n]


def get_segment_sentmeans(arc: list[float]) -> list[float]:
    """
    get the mean sentiment for each of the 20 segments of a sentiment arc (list of floats).
    """
    n_seg_items = len(arc) // 20
    segments = divide_segments(arc, n_seg_items)

    segment_means = [np.mean(segment) for segment in segments]
    return segment_means


def get_basic_sentarc_features(arc: list[float], length: int):
    """
    calculates basic features of the sentiment arc.
    """
    # basic features
    if length < 2:
        mean_sent = np.mean(arc)
        return mean_sent
    elif length < 10:
        mean_sent = np.mean(arc)
        std_sent = np.std(arc)
        return (mean_sent, std_sent)
    elif length < 20:
        mean_sent = np.mean(arc)
        std_sent = np.std(arc)
        # mean of first 10%, mean of last 10%
        n_ten_items = len(arc) // 10

        mean_first_ten = np.mean(arc[:n_ten_items])
        mean_end_ten = np.mean(arc[-n_ten_items:])

        # difference between end 10% and the rest
        mean_rest = np.mean(arc[:-n_ten_items])
        diff_end_rest = mean_rest - mean_end_ten

        return (mean_sent, std_sent, mean_first_ten, mean_end_ten, diff_end_rest)
    else:
        mean_sent = np.mean(arc)
        std_sent = np.std(arc)
        # mean of first 10%, mean of last 10%
        n_ten_items = len(arc) // 10

        mean_first_ten = np.mean(arc[:n_ten_items])
        mean_end_ten = np.mean(arc[-n_ten_items:])

        # difference between end 10% and the rest
        mean_rest = np.mean(arc[:-n_ten_items])
        diff_end_rest = mean_rest - mean_end_ten
        segment_means = get_segment_sentmeans(arc)

        return (
            mean_sent,
            std_sent,
            mean_first_ten,
            mean_end_ten,
            diff_end_rest,
            segment_means,
        )

def integrate(x: list[float]) -> np.matrix:
    return np.mat(np.cumsum(x) - np.mean(x))

def get_hurst(arc: list[float]):
    y = integrate(arc)
    uneven = y.shape[1] % 2
    if uneven:
        y = y[0, :-1]

    step_size = 1
    q = 3
    order = 1
    xy = md.multi_detrending(y, step_size, q, order)

    x = np.squeeze(np.asarray(xy[0]))
    y = np.squeeze(np.asarray(xy[1]))

    hurst = round(np.polyfit(x, y, 1)[0], 2)
    return hurst

def text_readability(text: str):
    flesch_grade = textstat.flesch_kincaid_grade(text)
    flesch_ease = textstat.flesch_reading_ease(text)
    smog = textstat.smog_index(text)
    ari = textstat.automated_readability_index(text)
    dale_chall_new = textstat.dale_chall_readability_score_v2(text)

    return flesch_grade, flesch_ease, smog, ari, dale_chall_new


def make_dico(lexicon: list) -> dict:
    tabs = [line.split("\t") for line in lexicon]

    words = [word[0] for word in tabs if len(tabs) > 1]
    counts = [word[1:] for word in tabs if len(tabs) > 1]

    dico = {}
    for i, word in enumerate(words):
        dico[word] = counts[i]

    return dico