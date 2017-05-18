#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pyspark

sc = pyspark.SparkContext('local[*]')

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

# Load and parse the data
data = sc.textFile("file:///home/jovyan/work/_data/cf_all.data")
ratingsTest = data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

data = sc.textFile("file:///home/jovyan/work/_data/cf_new.data")
ratingsNew = data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

ratingsAll=ratingsTest.union(ratingsNew)

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratingsAll, rank, numIterations)

# Evaluate the model on training data
testdata = ratingsAll.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratingsAll.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

itemNew=map(lambda x:x[1],ratingsNew.collect())
newUserId=ratingsNew.map(lambda x:x[0]).distinct().take(1)[0]
newUnrated=(predictions.filter(lambda x:x[0][1] not in itemNew).map(lambda x:(newUserId,x[0][1]))).distinct()
newRecommand= model.predictAll(newUnrated)
def g(x):
    print (x)
newRecommand.sortBy(lambda x: x[2],False).foreach(g)
rec=newRecommand.sortBy(lambda x: x[2],False).take(3)
print ('\n'.join( [ str(m[1])+","+str(m[2])  for m in rec] ))

# # Save and load model
# model.save(sc, "target/tmp/myCollaborativeFilter")
# sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")