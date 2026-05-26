

'''
    Returns tuned BM25 Parameters for the squad dataset as (b, k1)
'''
def Squad_BM25_tuned():
    return (0.75, 1.0)

'''
    Returns tuned BM25 Parameters for the WikiMultiHop dataset as (b, k1)
'''
def WikiMultiHop_BM25_tuned():
    return (0.9, 1.0)

'''
    Returns tuned BM25 Parameters for the squad dataset as (b, k1)
'''
def BM25_default():
    return (0.75, 1.2)