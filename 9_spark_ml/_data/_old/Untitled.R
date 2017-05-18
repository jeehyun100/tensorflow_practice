setwd("/Dropbox/201708_Miracom_ML/_data/")
c=read.csv("sample_libsvm_data__.txt")
names(c)=c("x","y","label")
plot(c$x,c$y,col= c("red", "green3")[c$label+1])
