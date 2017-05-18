from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD 
data = [
    LabeledPoint(0.0, [0.0, 1.0]),
    LabeledPoint(1.0, [1.0, 0.0]),
]
lrm = LogisticRegressionWithSGD.train(sc.parallelize(data), iterations=10)
lrm.predict([1.0, 0.0])
lrm.predict([0.0, 1.0])
lrm.predict(sc.parallelize([[1.0, 0.0], [0.0, 1.0]])).collect()
lrm.clearThreshold()
lrm.predict([0.0, 1.0])