val lines = sc.textFile("file:///data/spark-1.6.0/README.md")
val pairs = lines.map(s => (s, 1))
val counts = pairs.reduceByKey((a, b) => a + b)


//
// val x = sc.parallelize(Array(("a", 1), ("b", 1), ("a", 1),("a", 1), ("b", 1), ("b", 1),("b", 1), ("b", 1)), 3)
// val y = x.reduceByKey((accum, n) => (accum + n))
// y.collect
// val y = x.reduceByKey(_ + _)
// y.collect
// def sumFunc(accum:Int, n:Int) =  accum + n
// val y = x.reduceByKey(sumFunc)
// y.collect
// for (r <- y.collect ) println(s"$r")

