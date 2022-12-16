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
#print("--------------------------------")
#print(questions[0].id,questions[0].title,questions[0].body,questions[0].accepted_answer_id,questions[0].answers)
questions = recommendation.preprocess_all_questions(questions, idf, w2v) # matrix transformation
javadoc = pickle.load(open('../data/javadoc_pickle_wordsegmented','rb')) # the pre-trained knowledge base of javadoc
javadoc_dict_classes = dict()
javadoc_dict_methods = dict()
recommendation.preprocess_javadoc(javadoc,javadoc_dict_classes,javadoc_dict_methods,idf,w2v) # matrix transformation
parent = dict() # In online mode, there is no need to remove duplicate question of the query


print ('Loading data successful!')

mrr = 0.0
tot = 0.0

#filey=open("qtion.txt","a")

#for i in range(125810):
while True:
    #print("question")
    #print(i)
    print("Enter the query:")
    query = input()
    query_words = WordPunctTokenizer().tokenize(query.lower())
    if query_words[-1] == '?':
        query_words = query_words[:-1]
   # print("\nQuery as tokens:",query_words,"\n")
    query_words = [SnowballStemmer('english').stem(word) for word in query_words]
    #print("\nQuery after stemming:",query_words,"\n")

    query_matrix = similarity.init_doc_matrix(query_words, w2v)
    query_idf_vector = similarity.init_doc_idf_vector(query_words, idf)
    
    #print("The query idf vectors for each word:\n")
    #for ct,i in enumerate(query_words):
     #   print(i,"--->",query_idf_vector[0][ct],"\n")

    top_questions = recommendation.get_topk_questions(query, query_matrix, query_idf_vector, questions, 50, parent)
    recommended_api = recommendation.recommend_api(query_matrix, query_idf_vector, top_questions, questions, javadoc,javadoc_dict_methods,-1)
    

    pos = -1
    #val=""
    #val+=query
    for i,api in enumerate(recommended_api):
        #val+="$$$$$"+api
        print ('Rank',i+1,':',api)
        recommendation.summarize_api_method(api,top_questions,questions,javadoc,javadoc_dict_methods)
        if i==4:
            break
    #val+="\n"
    #filey.write(val)


    # query = 'Java Fastest way to read through text file with 2 million lines?'
    # query = 'How to round a number to n decimal places in Java'
    # query = 'run linux commands in java code'
    # query = 'How to remove single character from a String'
    # query = 'How to initialise an array in Java with a constant value efficiently'
    # query = 'How to generate a random permutation in Java?'
