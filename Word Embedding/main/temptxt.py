import _pickle as pickle
time_list = [0]
fd = open('word_embed_time.pickle','wb')
pickle.dump(time_list, fd)
fd.close()