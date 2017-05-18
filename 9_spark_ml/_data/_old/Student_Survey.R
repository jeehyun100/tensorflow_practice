set.seed(1234)
#모델을 생성하기 위한 집단과 평가를 위한 집단을 분리하기 위해서
#install.packages("party")
library(party)
setwd("/Dropbox/201708_Miracom_ML/_data/")
stu=read.csv("Student_Survey.csv",header=T ,fileEncoding = "EUC-KR")

myFormula <- 성별~필통+충전기+텀블러+노트북+책+파우치+지갑+안경집+치솔셋트+핸드크림+우산+향수+해드셋+차키+집키

stu_ctree <- ctree(myFormula, data=stu)
#결과 출력
print(stu_ctree)
plot(stu_ctree)
plot(iris_ctree,type="simple")

#평가
testPred <- predict(iris_ctree, newdata = testData)
table(testPred, testData$Species)
