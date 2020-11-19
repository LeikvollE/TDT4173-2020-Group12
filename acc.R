#!/usr/bin/env Rscript
d = read.csv("output/foo.csv", header = TRUE)
par(cex.axis=0.8, mar=c(8, 4, 5, 2))
boxplot(d, las=2)
