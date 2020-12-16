"""
data:{coauthorship, coauthor}
dataset:{cora, citeseer, pubmed}
"""
problem = 'coauthorship'
dataset = 'cora'
datasetroot = '../data/' + problem + '/' + dataset + '/'

"""
Configuration of the Network
num_class = {cora: 7, citeseer: }
"""
hidden_dim = 400
out_dim = 200
num_class = 7


"""
For training
"""
update_ratio = 0.004
seed = None
refit = 0
