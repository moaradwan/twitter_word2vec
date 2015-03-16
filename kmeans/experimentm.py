import csv
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk import pos_tag
from gensim.models.word2vec import Word2Vec
from gensim import matutils
from numpy import array, float32 as REAL
from sklearn.cluster import MiniBatchKMeans, KMeans
from multiprocessing import Pool




#string.punctuation
#string.digits


file = 'training.1600000.processed.noemoticon2.csv'
#file = 'testdata.manual.2009.06.14.csv'
tags = ["NNP", "NN", "NNS"]


ncls = 1000
niters = 1000
nreplay_kmeans = 1
lower = False

redundant = ["aw", "aww", "awww", "awwww", "haha", "lol", "wow", "wtf", "xd", "yay", "http", "www", "com", "ah", "ahh", "ahhh", "amp"]

def preprocess(tweet):
	ret_tweet = ""
	i = -1
	nn = []
	raw_tweet = tweet
	

	for ch in string.punctuation.replace("'","") + string.digits:
		tweet = tweet.replace(ch, " ")
	
	tweet_pos = {}
	if lower:
		tweet = tweet.lower()
	
	
	try:
		toks = word_tokenize(tweet)
		pos = pos_tag(toks)
		nn = [p for p in pos if p[1] in tags]
		#nn = [p for p in pos if p == 'NNP']
	except:
		pass
	if(len(nn)):
		tweet_pos["NN"] = nn
		ret_tweet = tweet_pos

			
	return ret_tweet


raw = []
with open(file, 'rb') as csvfile:
		content = csv.reader(csvfile, delimiter=',', quotechar='"')
		for row in content:
			tweet = row[5]
			raw.append(tweet)



p = Pool(6)
tweets = p.map(preprocess, raw)
t1 = []
t2 = []
for i in range(len(tweets)):
	if len(tweets[i]):
		t1.append(raw[i])
		t2.append(tweets[i])
raw = t1
tweets = t2


print "Loading model..."
wv = Word2Vec.load_word2vec_format('/home/attia42/workspace/projects/word2vec/data/GoogleNews-vectors-negative300.bin', binary=True)

vectors = []


for i in range(len(tweets)):
	tweet = tweets[i]
	nns = tweet['NN']
	vector = []
	#print nns 
	mean = []
	no_wv_tweet = True
	
	for w in nns:
		if len(w[0]) > 1 and w[0] in wv and w[0].lower() not in redundant:
			no_wv_tweet = False
			#print w[0]
			weight = 1
			if w[1] == 'NNP':
				weight = 100
			mean.append(weight * wv[w[0]])
	if(len(mean)):
		vectors.append(matutils.unitvec(array(mean).mean(axis=0)).astype(REAL))
	else:
		vectors.append([])




t1 = []
t2 = []
t3 = []
for i in range(len(vectors)):
	if vectors[i] != None and len(vectors[i]):
		t1.append(raw[i])
		t2.append(tweets[i])
		t3.append(vectors[i])
raw = t1
tweets = t2
vectors = t3




#kmeans = KMeans(init='k-means++', n_clusters=ncls, n_init=1)
kmeans = MiniBatchKMeans(init='k-means++', n_clusters=ncls, n_init=nreplay_kmeans, max_iter=niters)
kmeans.fit(vectors)


clss = kmeans.predict(vectors)

clusters = [[] for i in range(ncls)]

for i in range(len(vectors)):
	cls = clss[i]
	clusters[cls].append(i)


clusterstags = [[] for i in range(ncls)]

from collections import Counter

countarr = []
for c in clusters:
	counts = Counter()
	for i in c:
		t = [x[0] for x in tweets[i]["NN"] ]#if x[1] == "NNP"]
		#tn = [x[1] for x in tweets[i]["NN"]]
		sentence = " ".join(t) #+ tn)
		counts.update(word.strip('.,?!"\'').lower() for word in sentence.split())
	countarr.append(counts)




output = ""
for i in range(ncls):
	output = "Most common words for this cluster:\n"
	output += str(countarr[i].most_common(12))
	output += "\n\n\n\n\n\n"
	output += "Word2vec space of related words:\n"
	wv_rel = wv.most_similar([kmeans.cluster_centers_[i]], topn=10)
	output += str(wv_rel)
	output += "\n\n\n\n\n\n"
	for t in clusters[i]:
		output += str(raw[t]) + "\n"
	#output += "\n\n\n"
	nm = [x[0] for x in countarr[i].most_common(5)]
	nm = str(" ".join(nm)) 
	for ch in string.punctuation:
		nm = nm.replace(ch, " ")
	f = open('clusters/' + nm +'.txt', 'wb')
	f.write(output)
	f.close()




