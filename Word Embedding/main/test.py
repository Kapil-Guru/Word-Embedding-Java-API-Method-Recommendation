import recommendation
import read_data
from lxml import etree
from nltk.stem import SnowballStemmer
import similarity
from nltk.tokenize import WordPunctTokenizer
import gensim
import _pickle as pickle
from bs4 import BeautifulSoup
import util
import time


w2v = gensim.models.Word2Vec.load('../data/w2v_model_stemmed') # pre-trained word embedding
idf = pickle.load(open('../data/idf','rb')) # pre-trained idf value of all words in the w2v dictionary
questions = pickle.load(open('../data/api_questions_pickle_new', 'rb')) # the pre-trained knowledge base of api-related questions (about 120K questions)
questions = recommendation.preprocess_all_questions(questions, idf, w2v) # matrix transformation
javadoc = pickle.load(open('../data/javadoc_pickle_wordsegmented','rb')) # the pre-trained knowledge base of javadoc
javadoc_dict_classes = dict()
javadoc_dict_methods = dict()
recommendation.preprocess_javadoc(javadoc,javadoc_dict_classes,javadoc_dict_methods,idf,w2v) # matrix transformation
parent = pickle.load(open('../data/parent', 'rb')) # parent is a dict(), which stores the ids of each query's duplicate questions


querys = read_data.read_querys_from_file()
#querys = querys[:10]

print('loading data finished')


mrr = 0.0
map = 0.0
tot = 0
time_taken = 0
count = 0
time_list = []
for item in querys:
    count+=1
    #query = item[0].title
    start = time.time()
    query = item[0]
    true_apis = item[1]

    query_words = WordPunctTokenizer().tokenize(query.lower())
    query_words = [SnowballStemmer('english').stem(word) for word in query_words]

    query_matrix = similarity.init_doc_matrix(query_words, w2v)
    query_idf_vector = similarity.init_doc_idf_vector(query_words, idf)
    top_questions = recommendation.get_topk_questions(query, query_matrix, query_idf_vector, questions, 50, parent)
    recommended_api = recommendation.recommend_api(query_matrix, query_idf_vector, top_questions, questions, javadoc,javadoc_dict_methods,-1)


    #recommended_api = recommendation.recommend_api_baseline(query_matrix,query_idf_vector,javadoc,-1)


    pos = -1
    tmp_map = 0.0
    hits = 0.0
    for i,api in enumerate(recommended_api):
        if api in true_apis and pos == -1:
            pos = i+1
        if api in true_apis:
            hits += 1
            tmp_map += hits/(i+1)
    tmp_map /= len(true_apis)
    tmp_mrr = 0.0
    if pos!=-1:
        tmp_mrr = 1.0/pos

    map += tmp_map
    mrr += tmp_mrr

    end = time.time()
    time_taken += (end - start)
    time_list.append(end-start)

    print(count, time_taken)


    # for i, api in enumerate(recommended_api):
    #     if i==10:
    #         break
    #     print api,'rank',i
    #     recommendation.summarize_api_method(api,top_questions,questions,javadoc,javadoc_dict_methods)


avg_time_taken = time_taken / len(querys)
print('Mean Reciprocal Rank:',mrr/len(querys))
print('Total queries used for testing:',len(querys))
print('Mean Average Precision',map/len(querys))
print('Average time taken for a query',avg_time_taken)
fd = open('word_embed_time.pickle','wb')
pickle.dump(time_list, fd)
fd.close()