from scipy.stats import spearmanr
from scipy.spatial.distance import squareform


def sq(x):
    return squareform(x, force='tovector', checks=False)


def get_corr(m1, m2):
    return spearmanr(sq(m1), sq(m2))[0]
