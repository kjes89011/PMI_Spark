# PMI_Spark
This is a Spark version of Pointwise mutual information

Usage
=============

1. Put Input file into PMIInput directory. One line per document, and each word is separated by spaces.
2. Rename input file to "CleanCorpus.txt"
3. Run `Spark-submit PMI.py`.
4. In the code, you can choose to save the OutputFile in HDFS or write a file.
