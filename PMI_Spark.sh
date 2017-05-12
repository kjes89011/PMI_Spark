hadoop fs -put $1 $1
spark-submit PMI.py $1
hadoop fs -cat Word_PMI_Similarity/part* > $2
rm $1
hadoop fs -rm $1
hadoop fs -rm -r Word_PMI_Similarity

