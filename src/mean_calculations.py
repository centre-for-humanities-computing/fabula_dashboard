# script for reading in .xlsx files and calculating the mean of the columns with numerical data

# read in the file
import pandas as pd
df = pd.read_excel('data/CHICAGO_MEASURES_FEB24.xlsx')

# calculate the mean of the columns with numerical data
# identify the columns with numerical data
#numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
# calculate the mean of the columns with numerical data
#mean = df[numerical_columns].mean()
df_subset = df.loc[:,['TITLE_LENGTH', 'HURST', 'WORDCOUNT', 
                       'SENTENCE_LENGTH', 'BZIP_NEW', 'MSTTR-100', 
                       'BZIP_TXT', 'READABILITY_FLESCH_GRADE', 
                       'READABILITY_FLESCH_EASE', 'READABILITY_SMOG',
                       'READABILITY_ARI', 'READABILITY_DALE_CHALL_NEW',
                       'MEAN_SENT', 'STD_SENT', 'END_SENT',
                       'BEGINNING_SENT', 'DIFFERENCE_ENDING_TO_MEAN', 
                    #    'ARC_SEGMENTS_MEANS',
                       'SPACY_ADJ', 'SPACY_NOUN',
                       'SPACY_VERB','SPACY_ADV', 'SPACY_PRON',
                       'SPACY_PUNCT', 'SPACY_STOPS', 'SPACY_SBJ', 'SPACY_PASSIVE',
                       'SPACY_NSUBJ', 'SPACY_AUX', 'SPACY_RELATIVE', 
                       'SPACY_NEGATION', 'BIGRAM_ENTROPY','WORD_ENTROPY',
                       'AVG_WORDLENGTH','bigram_entropy','word_entropy',
                       'ang_slopes','ang_skews','ang_kurtoses','ang_overall',
                       'dis_slopes','dis_skews','dis_kurtoses','dis_overall',
                       'fea_slopes','fea_skews','fea_kurtoses','fea_overall',
                       'ant_slopes','ant_skews','ant_kurtoses','ant_overall',
                       'sur_slopes','sur_skews','sur_kurtoses','sur_overall',
                       'sad_slopes','sad_skews','sad_kurtoses','sad_overall',
                       'joy_slopes','joy_skews','joy_kurtoses','joy_overall',
                       'HURST_SYUZHET','TTR_VERB','TTR_NOUN','FREQ_OF','FREQ_THAT',
                       'self_model_ppl','gpt2_ppl','gpt2-xl_ppl','VERB_NOUN_RATIO',
                       'ADV_VERB_RATIO','PERC_ACTIVE_VERBS','PASSIVE_ACTIVE_RATIO',
                       'NOMINAL_VERB_RATIO','APPENT_SYUZHET','mean_con','mean_val',
                       'mean_aro','mean_dom','std_con','std_val','std_aro','std_dom']]

# change column names to be more descriptive

df_subset.columns = ['TITLE_LENGTH', 'HURST', 'word_count', 
                       'SENTENCE_LENGTH', 'BZIP_NEW', 'MSTTR-100', 
                       'BZIP_TXT', 'flesch_grade', 
                       'flesch_ease', 'smog',
                       'ari', 'dale_chall_new',
                       'MEAN_SENT', 'STD_SENT', 'END_SENT',
                       'BEGINNING_SENT', 'DIFFERENCE_ENDING_TO_MEAN', 
                    #    'ARC_SEGMENTS_MEANS',
                       'SPACY_ADJ', 'SPACY_NOUN',
                       'SPACY_VERB','SPACY_ADV', 'SPACY_PRON',
                       'SPACY_PUNCT', 'SPACY_STOPS', 'SPACY_SBJ', 'SPACY_PASSIVE',
                       'SPACY_NSUBJ', 'SPACY_AUX', 'SPACY_RELATIVE', 
                       'SPACY_NEGATION', 'BIGRAM_ENTROPY','WORD_ENTROPY',
                       'AVG_WORDLENGTH','bigram_entropy','word_entropy',
                       'ang_slopes','ang_skews','ang_kurtoses','ang_overall',
                       'dis_slopes','dis_skews','dis_kurtoses','dis_overall',
                       'fea_slopes','fea_skews','fea_kurtoses','fea_overall',
                       'ant_slopes','ant_skews','ant_kurtoses','ant_overall',
                       'sur_slopes','sur_skews','sur_kurtoses','sur_overall',
                       'sad_slopes','sad_skews','sad_kurtoses','sad_overall',
                       'joy_slopes','joy_skews','joy_kurtoses','joy_overall',
                       'HURST_SYUZHET','TTR_VERB','TTR_NOUN','FREQ_OF','FREQ_THAT',
                       'self_model_ppl','gpt2_ppl','gpt2-xl_ppl','VERB_NOUN_RATIO',
                       'ADV_VERB_RATIO','PERC_ACTIVE_VERBS','PASSIVE_ACTIVE_RATIO',
                       'NOMINAL_VERB_RATIO','APPENT_SYUZHET','mean_con','mean_val',
                       'mean_aro','mean_dom','std_con','std_val','std_aro','std_dom']

mean = df_subset.mean()
# save the mean to a new dataframe with the old column names and one row wth the mean
mean_df = pd.DataFrame(mean, columns=['Mean'])
# transpose the dataframe
mean_df = mean_df.T

# save the mean dataframe to a csv file
mean_df.to_csv('data/mean.csv')
