from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]   # 모든 값을 실수로
    return LabeledPoint(values[0], values[1:])

data = sc.textFile("file:///home/hadoop/2.data")
parsedData = data.map(parsePoint)

# Build the model
model = LogisticRegressionWithLBFGS.train(parsedData)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count()) # 실제값과 예측값이 틀린것 갯수 count
print("Training Error = " + str(trainErr))

# Save and load model
# model.save(sc, "target/tmp/pythonLogisticRegressionWithLBFGSModel")
# sameModel = LogisticRegressionModel.load(sc,"target/tmp/pythonLogisticRegressionWithLBFGSModel")