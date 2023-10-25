import hazm
import parsivar
from string import punctuation
import json
import re

class DataPreprocessing:
    
    #Tokenization
    @staticmethod
    def Tokenization(text):
        return hazm.word_tokenize(text)
    
    #Normalization
    @staticmethod
    def Normalization(text):
        my_normalizer = hazm.Normalizer()
        return my_normalizer.normalize(text)
    
    #Stop_Word_Removal
    @staticmethod
    def Stop_Word_Removal(tokens):
        stop_words = hazm.stopwords_list()
        tokens_with_removed_stopwords = []
        for token in tokens:
            if not (token in stop_words):
                tokens_with_removed_stopwords.append(token)
        return tokens_with_removed_stopwords
    
    #Stemming
    @staticmethod
    def Stemming(tokens):
        stemmed = []
        my_stemmer = parsivar.FindStems()
        for token in tokens:
            stemmed.append(my_stemmer.convert_to_stem(token))
        return stemmed
    
    #Remove Punctuations
    @staticmethod
    def Remove_Punctuations(text):
        return re.sub(f'[{punctuation}؟،٪×÷»«]+', '', text)

    def preprocess(self, docs, contents):
        for idx, content in enumerate(contents):
            punctuated_content = self.Remove_Punctuations(content)
            normalized_content = self.Normalization(punctuated_content)
            tokens_of_a_sentence = self.Tokenization(normalized_content)
            tokens_of_a_sentence = self.Stop_Word_Removal(tokens_of_a_sentence)
            final_tokens_of_a_sentence = self.Stemming(tokens_of_a_sentence)
            docs[str(idx)]['content'] = final_tokens_of_a_sentence
            if idx % 1000 == 0:
                print(idx)
        return docs
    
def Load_Docs(): 

    all_docs = {}
    contents = []
    urls = []

    with open("IR_data_news_12k.json", 'r') as f:
        docs = json.load(f)
        for k in docs.keys():
            # index of files
            idx = k + '' 
            all_docs[idx] = {'title': docs[idx]['title'],
                             'content': docs[idx]['content'],
                             'url': docs[idx]['url'],
                            }
            contents.append(docs[idx]['content'])
    return all_docs, contents, urls

def Postings_List(Docs):
    my_dict = {}
    for index in Docs:
        for position, token in enumerate(Docs[index]['content']):
            
            if token in my_dict:
                
                if index in my_dict[token]['docs']:
                    my_dict[token]['docs'][index]['positions'].append(position)
                    my_dict[token]['docs'][index]['number_of_token'] += 1 
                else:
                    my_dict[token]['docs'][index] = { 
                                                    'positions': [position],
                                                    'number_of_token': 1
                                                    }
                my_dict[token]['frequency'] += 1
            else:
                my_dict[token] = {
                 'frequency': 1,
                 'docs': {
                       index: {
                           'pos': [position],
                           'number_of_token': 1
                           }
                    }
                }
    return my_dict

if __name__ == "__main__" :
    #Load Docs
    all_docs, contents, urls = Load_Docs()
    with open("IR_data_news_12k.json") as f:
        docs = json.load(f)
    #Preprocessing
    preprocessor = DataPreprocessing()
    pre_processed_docs = preprocessor.preprocess(docs, contents)
    #Write Preprocessed Docs
    with open("IR_data_news_12k_preprocessed.json", 'w') as f:
        json.dump(pre_processed_docs, f)
    dictionary = Postings_List(pre_processed_docs)
    print(dictionary['فارس'])




    
