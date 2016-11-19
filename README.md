# Use Apache Spark 2.0.2 to optimize t-sne parameters

The best t-sne mapping is the one minimizing the Kullback-Leibler divergence.

Using Spark we parallelize this operation.

# Script parameters
Input file, number of dimensions, output pickle path.
