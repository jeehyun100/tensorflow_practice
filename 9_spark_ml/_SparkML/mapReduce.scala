val lines = sc.textFile("file:///data/spark-1.6.0/README.md")
val lineLengths = lines.map(s => s.length)
val totalLength = lineLengths.reduce((a, b) => a + b)
totalLength // 글자 갯수 파악
