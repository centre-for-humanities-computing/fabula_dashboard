import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from lexical_diversity import lex_div as ld
import neurokit2 as nk

from utils import *


def compute_metrics(text:str, lang:str, sentiment_method:str) -> dict:

    # download nltk resources
    nltk.download("punkt")
    nltk.download("wordnet")

    # create lemmatizer
    lmtzr = WordNetLemmatizer()

    # load spacy model according to language
    nlp = get_nlp(lang)
    nlp.max_length = 3500000

    # dict for saving metrics
    output = {}

    # prepare text and tokens
    sents = sent_tokenize(text, language=lang)
    words = word_tokenize(text, language=lang)

    # spacy
    spacy_attributes = []
    for token in nlp(text):
        token_attributes = get_spacy_attributes(token)
        spacy_attributes.append(token_attributes)

    spacy_df = create_spacy_df(spacy_attributes)

    # stylometrics
    ## for words
    output["word_count"] = len(words)
    output["average_wordlen"] = avg_wordlen(words)
    output["msttr"] = ld.msttr(words, window_length=100)

    # for sentences
    if len(sents) < 1502:
        print("text not long enough for stylometrics\n")

    else:
        output["average_sentlen"] = avg_sentlen(sents)
        output["gzipr"], output["bzipr"] = compressrat(sents)

    # bigram and word entropy
    try:
        output["bigram_entropy"], output["word_entropy"] = text_entropy(text, language=lang, base=2, asprob=False)
    except:
        print("error in bigram and/or word entropy\n")

    # setting up sentiment analyzer
    if "vader" in sentiment_method:
        nltk.download("vader_lexicon")
    
    arc = get_sentarc(sents, sentiment_method, lang)

    # basic sentiment features
    if len(arc) < 60:
        print("arc not long enough for basic sentiment features\n")

    else:
        (
            output["mean_sentiment"],
            output["std_sentiment"],
            output["mean_sentiment_per_segment"],
            output["mean_sentiment_first_ten_percent"],
            output["mean_sentiment_last_ten_percent"],
            output["difference_lastten_therest"],
        ) = get_basic_sentarc_features(arc)

    # approximate entropy
    try:
        output["approximate_entropy"] = nk.entropy_approximate(
            arc, dimension=2, tolerance="sd"
        )
    except:
        print("error with approximate entropy\n")
    
    # hurst
    try:
        output["hurst"] = get_hurst(arc)
    except:
        print("error with hurst\n")

    # doing the things that only work in English
    if lang == "english":
        # readability
        try:
            (
                output["flesch_grade"],
                output["flesch_ease"],
                output["smog"],
                output["ari"],
                output["dale_chall_new"],
            ) = text_readability(text)

        except:
            print("error in readability\n")

        # concreteness and VAD
        diconc = json.load("data/concreteness_dict.json")

        with open("NRC-VAD-Lexicon.txt", "r") as f: # not in fabula_pipeline repo?
            lexicon = f.readlines()

        dico = make_dico(lexicon)

        conc = []
        val, aro, dom = [], [], []

        for sent in sents:
            words = word_tokenize(sent)
            lemmas = [lmtzr.lemmatize(word) for word in words]

            for lem in lemmas:
                if lem in diconc.keys():
                    conc.append([diconc[lem]])
                if lem in dico.keys():
                    val.append([dico[lem][0]])
                    aro.append([dico[lem][1]])
                    dom.append([dico[lem][2]])

        output["concreteness"] = conc
        output["valence"] = val
        output["arousal"] = aro
        output["dominance"] = dom

        # roget
        all_roget_categories = roget.list_all_categories()

        roget_df = filter_spacy_df(spacy_df)

        output["roget_n_tokens"] = len(spacy_df)
        output["roget_n_tokens_filtered"] = len(roget_df)

        token_categories = get_token_categories(roget_df)
        doc_categories = re.findall(r"(rog\d{3} \w*)", token_categories)

        for roget_cat in all_roget_categories:
            output[roget_cat] = doc_categories.count(roget_cat)

        output["roget_n_cats"] = len(doc_categories)

        # save arc
        output["arc"] = arc

        # return the metrics
        return output
