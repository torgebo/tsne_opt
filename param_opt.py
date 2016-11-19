"""
Calculating optimal parameters for t-sne can be time consuming.
This small script uses sampling and Apache Spark v.2.0.2 for parallelism.
"""
import sys
import pandas
import itertools 
import gc
import pickle

from pprint import pprint
from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np


from pyspark import SparkConf, SparkContext, StorageLevel



def read_inp(path_in):
    csv = pandas.read_csv(path_in) 
    matrix = np.array(csv.ix[:,1:].copy())
    return matrix
 

def generate_params():
    perplexities = list(set(np.random.random_integers(10, 100, 1)))
    learning_rates = list(set(np.random.random_integers(10, 100, 4)))
    random_states = list(range(4))
    return [_ for _ in itertools.product(perplexities, learning_rates, random_states)]


def fit_pred(matrix, *args, **kwargs):
    """ Fit TSNE model to matrix with given arguments """
    try:
        model = TSNE(*args, **kwargs)
        y = model.fit_transform(matrix)
    except Exception as e:
        return math.inf, None, None
    return model.kl_divergence_, model, y


class ArgIter:

    def __init__(self, in_iter, matrix, n_comp):
        self.in_iter = in_iter
        self.matrix = matrix
        self.n_comp = n_comp

    def __iter__(self):
        return self

    def __next__(self):
        perp, rate, state = self.in_iter.__next__()
        kl, _, _ = fit_pred(
            self.matrix,
            n_components=self.n_comp,
            perplexity=perp,
            learning_rate=rate,
            random_state=state
        )
        gc.collect()
        return kl, perp, rate, state


def iter_fit(bma, bm_comp):
    """
    Distribute a an iterable function
    containin matrix for learning
    and number of components
    (as broadcast variables).
    """
    matrix = bma.value
    n_components = bm_comp.value

    def iterate(iterator):
        argiter = ArgIter(iterator, matrix, n_components)
        return argiter

    return iterate


def optimize(rdd, bma, bm_comp):
    """
    Arguments:
    ----------
    rdd: RDD<int, int, int>
        the parameter space, each row being
        (perplexity, learning_rate, random_state)
    bma: broadcast variable<np.array> 
        of input manifold matrix
    bm_comp: broadcast variable<int>
        output manifold dimension
    
    Returns:
    --------
    RDD<float, int, int, int>
        (kl_divergence, perplexity, learning_rate, random_state)
    """
    return rdd.mapPartitions(iter_fit(bma, bm_comp))


def main(path_in, n_components, output_path):
    """
    Arguments:
    ----------
    path_in: str
        path to csv file defining principal manifold
    n_components: int
        dimensionality of ouput manifold. 2 or 3 for visualization.
    output_path:
        pickle output_path for collected parameters.
    """

    matrix = read_inp(path_in)
        
    conf = SparkConf().setAppName("tsne_opt")
    sc = SparkContext(conf=conf)


    bm_ma = sc.broadcast(matrix)
    bm_comp = sc.broadcast(n_components)

    params = generate_params()
    print("Params are:")
    pprint(params)

    
    paraRDD = sc.parallelize(params)

    # Estimate parameter cost iteratively on each partition
    rdd_opt_params = optimize(paraRDD, bm_ma, bm_comp)
    
    kl_params = rdd_opt_params.collect()
    sc.stop()

    print("Parameters found ( ) are")
    pprint(kl_params)
    
    
    with open(output_path, "wb") as f:
        pickle.dump(kl_params, f)
    
    print("Finished")
    

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("""
        Provide arguments to sys.argv (in order):
        path_in:
        path to csv file containing matrix
        n_components: 
        number of dimensions for embedding space,
        usually 2 or 3.
        output_path:
        path of pickled arguments object.
        """)
        sys.exit(1)
    
    path_in = sys.argv[1]
    n_components = int(sys.argv[2])
    output_path = sys.argv[3]
    print("Using path_in={}".format(path_in))
    print("Using n_components={}".format(n_components))
    print("Using output_path={}".format(output_path))
    
    main(path_in, n_components, output_path)
