val textFile = sc.textFile("file:///data/spark-1.6.0/README.md")
//val counts = textFile.flatMap(line => line.split(" ")).map(word => (word, 1)).reduceByKey(_ + _)
val counts = textFile.flatMap(line => line.split(" "))
            .map(_.toUpperCase())
            .map(_.replace("\""," "))
            .map(word => (word, 1))
            .reduceByKey(_+_)
            .sortByKey()
            .filter(r=> r._1 !="")
            //.foreach(println)
counts.saveAsTextFile("file:///home/hadoop/wc")
