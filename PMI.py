#-*- coding: utf-8 -*-
from pyspark import SparkContext, SparkConf
import math,time,sys

start_time = time.time()

conf = SparkConf().setAppName("PMI")
sc = SparkContext(conf = conf)
window_size = 5	#左右各5個詞

def window(origin):
	origin = origin.split(" ")
	output = []
	for Hitword in origin:
		index = origin.index(Hitword)	
		start = (index - 5) if (index - 5) > 0 else 0
		leng = len(origin) - start
		if leng >= 11:	#array夠長，可以直接11個長度
			tmp = origin[start:(start + window_size * 2 + 1)]		
		else:	#不夠11的，則是剩餘的長度
			tmp = origin[start:]
		tmp = list(set(tmp))
		for x in tmp:
			if x == Hitword:
				continue
			output.append(((Hitword,x),1))
	return output

def log2(number):
	log2 = math.log(number)/math.log(2)
	return log2

#將WordNeighbors與其對應的wordCount組成(WordNeighbors,wordCount)的pair
#Example of word : (u'consists', [(u'more', u'consists', '1'), (u'first', u'consists', '2'), (u'consists', 3)])
def PreProcessingPMI_part1(word):
	output = []
	for WNpair in word[1]:
		if len(WNpair) == 2:
			neighborWC = WNpair
			break
	for WNpair in word[1]:
		if len(WNpair) == 2:
			continue
		output.append((WNpair[0],[(WNpair,neighborWC)]))	
	return output

#取出Word與其對應的wordCount組成(Word,wordCount)的pair
#Example of word : #(u'nutritious', [((u'nutritious', u'delicious', '1'), (u'delicious', 18)), ((u'nutritious', u'remain', '1'), (u'remain', 2)), (u'nutritious', 1)])
def PreProcessingPMI_part2(word):
	output = []
	for WNpair in word[1]:
		if WNpair[0] == word[0]:
			WordWC = WNpair
			break
	for WNpair in word[1]:
		if WNpair[0] == word[0]:
			continue
		output.append((WordWC,)+WNpair)	
	return output

#Example of word : ((u'nutritious', 1), (u'nutritious', u'delicious', '1'), (u'delicious', 18))
def PMI(word):
	W_N = 1.0		#W_N 為 Word 與 Neighbor各自出現的次數
	for value in word:
		if len(value) == 3:
			WN = float(value[2])*NumberOfWordsInArticle	#WN 為 Word 與 Neighbor 共同出現的次數
			output = value
		else:
			W_N=W_N*float(value[1])
	PMI = log2(WN/W_N)
	output = (output[0],output[1],PMI)
	return output

#輸入原始文字檔
text_file = sc.textFile(sys.argv[1])

#WordCount
WordCounts = text_file.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

#文章共有幾個不同的詞
NumberOfWords = len(WordCounts.collect())

#文章總詞數
NumberOfWordsInArticle = WordCounts.map(lambda word:int(word[1])).reduce(lambda a,b:a+b)

#抓出 Word 的 Neighbor 以及共同出現次數
WordNeighbors = text_file.map(window).flatMap(lambda q:q).reduceByKey(lambda a,b:a+b).map(lambda q : (q[0][0],(q[0][1],str(q[1]))))

WordNeighborsKeyList = WordNeighbors.map(lambda a:a[0]).distinct().collect()

#開始計算PMI

#只取出 CandidateNeighborsKeyList 的 WordCount
WordNeighbors_WordCounts = WordCounts.filter(lambda word: word[0] in WordNeighborsKeyList).map(lambda word : (word[0],[word]))

#將 Neighbor 當 key 以透過 Reduce 取得 Neighbor 的 WordCount
NeighborCandidate = WordNeighbors.map(lambda q : (q[1][0],[(q[0],q[1][0],q[1][1])])).union(WordCounts.map(lambda word : (word[0],[word]))).reduceByKey(lambda a,b:a+b).filter(lambda word:len(word[1])>1)

pmi = NeighborCandidate.map(PreProcessingPMI_part1).flatMap(lambda line:line).union(WordNeighbors_WordCounts).reduceByKey(lambda a,b:a+b).map(PreProcessingPMI_part2).flatMap(lambda a:a).map(PMI).filter(lambda word: word[2] != 0)

#map from tuple to String and charset to Utf8
pmi = pmi.map(lambda word:u'\t'.join(unicode(s) for s in word).encode("utf-8").strip())

pmi.saveAsTextFile("Word_PMI_Similarity")

#輸出Debug用
# for x in pmi.take(10):
# 	print x

#f = open("./Word_PMI_Similarity.txt","w")
#for word in pmi.collect():
#	f.write(word + '\n')
#f.close()

sc.stop()

print("--- %s seconds ---" % (time.time() - start_time))
