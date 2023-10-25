import hazm
import parsivar
from string import punctuation
import json
import re
import numpy as np
import collections

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
    
    #Query preprocessing
    def simple_preprocess(self, content):
        punctuated_content = self.Remove_Punctuations(content)
        normalized_content = self.Normalization(punctuated_content)
        tokens_of_a_sentence = self.Tokenization(normalized_content)
        tokens_of_a_sentence = self.Stop_Word_Removal(tokens_of_a_sentence)
        final_tokens_of_a_sentence = self.Stemming(tokens_of_a_sentence)
            #if idx % 1000 == 0:
                #print(idx)
        return final_tokens_of_a_sentence

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
                           'positions': [position],
                           'number_of_token': 1
                           }
                    }
                }
    #Champions List
    for term in my_dict:
        term_docs = dict(my_dict[term]['docs'])
        sorted_term_docs = sorted(term_docs, key=lambda doc: term_docs[doc]['number_of_token'], reverse=True)
        my_dict[term]['champions_list'] = {}
        for doc_number in sorted_term_docs:
            my_dict[term]['champions_list'][doc_number] = {'number_of_token': my_dict[term]['docs'][doc_number]['number_of_token']}

    return my_dict

class QueryProcessing:
    
    @staticmethod
    def find_doc_relevance(tokens):
        docs_relevence = {}
        for token in tokens:
            if token in dictionary:
                for doc_id in dictionary[token]['docs']:
                    if doc_id in docs_relevence.keys():
                        docs_relevence[doc_id] += 1
                    else:
                        docs_relevence[doc_id] = 1

        docs_relevence = sorted(docs_relevence.items(), key=lambda x: x[1], reverse=True)
        final_docs = []
        for doc, count in docs_relevence:
            final_docs.append(doc)
        return final_docs
    
    @staticmethod
    def remove_exceptions_docs(docs, query):
        
        exp_tokens = re.findall(r'\!\s(\w+)', query)
        if len(exp_tokens):
            exp_docs = []
            print("exp", exp_tokens)
            for token in exp_tokens:
                for doc_id in dictionary[token]['docs'].keys():
                    if doc_id not in exp_docs:
                        exp_docs.append(doc_id)

            final_docs = []
            for doc_id in docs:
                if doc_id not in exp_docs:
                    final_docs.append(doc_id)
            return final_docs

    def phrase_process(self, docs, query):
        
        phrase_tokens = re.findall(r'"([^"]*)"', query)
        if len(phrase_tokens):
            lists = []
            final_docs = []
            for phrase in phrase_tokens:
                terms = phrase.split()
                for term in terms:
                    if term in dictionary:
                        lists.append(list(dictionary[term]['docs'].keys()))

                merge_lists = set.intersection(*map(set, lists))

                for doc_id in merge_lists:
                    if self.phrase_order(doc_id, phrase):     
                        final_docs.append(doc_id)
            return final_docs
        
    @staticmethod
    def phrase_order(doc_id, phrase):
        phrase_split = phrase.split()
        dic_pos = []
        for token in phrase_split:
            dic_pos.append(list(dictionary[token]['docs'][doc_id]['positions']))

        for first_idx in dic_pos[0]:
            check_order = True
            for j in range (len(dic_pos)):
                if first_idx + j not in dic_pos[j]:
                    check_order = False
            if check_order:
                return True
        return check_order
    
    def query_processing(self, query):
        # remove exception tokens: "!'space''word'"
        query_remove_exp = re.sub(r'\!\s\w+', '', query)
        #remove quotes
        query_tokens = re.sub(r'"[^"]*"', '', query_remove_exp)
        
        #query preprocessing
        query_preprocessor = DataPreprocessing()
        preprocessed_query_tokens = query_preprocessor.simple_preprocess(query_tokens)
        
        #finding query tokens in docs by relevance
        result = None
        if len(query_tokens) != 0:
            result = self.find_doc_relevance(preprocessed_query_tokens)
        
        #remove docs with exceptions tokens
        remove_exceptions_result = self.remove_exceptions_docs(result, query)
        if remove_exceptions_result != None and len(result) > 0:
            result = [r for r in result if r in remove_exceptions_result] 
        
        #find phrases
        phrase_result = self.phrase_process(pre_processed_docs, query)
        if result == None:
            return phrase_result
        if phrase_result != None:
            return [r for r in result if r in phrase_result]
        return result

def print_result(results, docs):
    for rank, doc_id in enumerate(results):
        if doc_id == None:
            continue
        print(100*'-')
        print(f'Rank: {rank + 1}, docID: {doc_id}')
        print(f'Title: {docs[f"{doc_id}"]["title"]}')
        print(f'URL: {docs[f"{doc_id}"]["url"]}')

def query_result(query, docs):
    
    query_processor = QueryProcessing()
    results = query_processor.query_processing(query)
    if(len(results) > 50):
        results = results[:5]
    if results == None:
        print("نتیجه ای یافت نشد")
    else:
        print_result(results, docs)

################################################################
#Phase 2
################################################################

def calculate_tf_idf(f_td, N, n_t):
    tf = 1 + np.log10(f_td)
    idf = np.log10(N / n_t)
    return tf * idf

def query_indexing(query):
    #query preprocessing
    query_preprocessor = DataPreprocessing()
    query_tokens = query_preprocessor.simple_preprocess(query)
    query_tokens_count = dict(collections.Counter(query_tokens))
    return query_tokens_count

def query_scoring(query, total_number_of_docs, dictionary, k, champion_list = False):
    
    doc_scores1 = [0 for i in range(0, total_number_of_docs)]
    doc_scores2 = [0 for i in range(0, total_number_of_docs)]
    query_tokens_count = query_indexing(query)
    
    query_length = sum(query_tokens_count.values())
    
    for term in query_tokens_count:
        if term in dictionary:
            if champion_list: 
                term_docs = dictionary[term]['champions_list']
            else:
                term_docs = dictionary[term]['docs']
            #calculate w_{t,q}
            w_tq = calculate_tf_idf(query_tokens_count[term], total_number_of_docs, len(term_docs))
            for doc in term_docs:
                #calculate w_{t,d}
                w_td = calculate_tf_idf(term_docs[doc]['number_of_token'], total_number_of_docs, len(term_docs))
                #update doc scores for cosines similarity
                doc_scores1[int(doc)] += w_td * w_tq
                #update doc scores for jaccard similarity
                doc_scores2[int(doc)] += 1
                
    for doc_number in range(len(doc_scores1)):
        doc_scores1[doc_number] /= len(pre_processed_docs[str(doc_number)]['content'])
    for doc_number in range(len(doc_scores2)):
        doc_scores2[doc_number] /= (len(pre_processed_docs[str(doc_number)]['content']) + query_length - doc_scores2[doc_number])
    
    sorted_doc_numbers1 = np.argsort(doc_scores1)
    sorted_doc_numbers1 = np.flip(sorted_doc_numbers1)
    
    sorted_doc_numbers2 = np.argsort(doc_scores2)
    sorted_doc_numbers2 = np.flip(sorted_doc_numbers2)
    
    return sorted_doc_numbers1[:k], sorted_doc_numbers2[:k]
        
def query_result_phase2(query, result_numbers = 5, champion_list = False):
    
    results1, results2 = query_scoring(query, len(docs), dictionary, result_numbers, champion_list)
    if len(results1) == 0 and len(results2) == 0:
        print("نتیجه ای یافت نشد")
    else:
        print("Cosine Scores:")
        print(100*'=')
        print_result(results1, pre_processed_docs)
        print()
        print("Jaccard Scores:")
        print(100*'=')
        print_result(results2, pre_processed_docs)

if __name__ == "__main__" :
    #Load Docs
    all_docs, contents, urls = Load_Docs()
    with open("IR_data_news_12k.json") as f:
        docs = json.load(f)
    #Preprocessing
    #preprocessor = DataPreprocessing()
    #pre_processed_docs = preprocessor.preprocess(docs, contents)
    #Write Preprocessed Docs
   # with open("IR_data_news_12k_preprocessed.json", 'w') as f:
        #json.dump(pre_processed_docs, f)

    # run second time
    global pre_processed_docs
    with open("IR_data_news_12k_preprocessed.json", 'r') as f:
        pre_processed_docs = json.load(f)
    #Creat Postings Lists
    global dictionary 
    dictionary = Postings_List(pre_processed_docs)
    #print(dictionary['فارس'])

    #Query Processing
    while(1):
        phase = (int)(input("Enter Query Processing Phase: "))
        if phase == 0:
                break
        if phase == 1:
            query = input("Enter Query: ")
            query_result(query, pre_processed_docs)
        elif phase == 2:
            print("hi")
            query = input("Enter Query: ")
            query_result_phase2(query, champion_list = True)
        










    
