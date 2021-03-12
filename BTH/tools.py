import numpy as np
import scipy.io as sio 
import scipy.sparse as sp


def load_data(path, dtype=np.float32):
    db = sio.loadmat(path)
    traindata = dtype(db['traindata'])
    testdata = dtype(db['testdata'])
    cateTrainTest = dtype(db['cateTrainTest'])

    mean = np.mean(traindata, axis=0)
    traindata -= mean
    testdata -= mean

    return traindata, testdata, cateTrainTest

def save_sparse_matrix(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape, _type=array.__class__)

def load_sparse_matrix(filename):
    matrix = np.load(filename)

    _type = matrix['_type']
    sparse_matrix = _type.item(0)

    return sparse_matrix((matrix['data'], matrix['indices'],
                                 matrix['indptr']), shape=matrix['shape'])

def binarize_adj(adj):
    adj[adj != 0] = 1
    return adj
        
def renormalize_adj(adj):
    rowsum = np.array(adj.sum(axis=1))
    inv = np.power(rowsum, -0.5).flatten()
    inv[np.isinf(inv)] = 0.
    zdiag = sp.diags(inv)     

    return adj.dot(zdiag).transpose().dot(zdiag)

def sign_dot(data, func):
    return np.sign(np.dot(data, func))

def mAP(cateTrainTest, IX, num_return_NN=None):
    numTrain, numTest = IX.shape

    num_return_NN = numTrain if not num_return_NN else num_return_NN

    apall = np.zeros((numTest, 1))
    yescnt_all = np.zeros((numTest, 1))
    for qid in range(numTest):
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1
                p += x/(rid*1.0 + 1.0)
	
	yescnt_all[qid] = x
        if not p: apall[qid] = 0.0
        else: apall[qid] = p/(num_return_NN*1.0)

    return np.mean(apall),apall,yescnt_all  

def topK(cateTrainTest, HammingRank, k=500):
    numTest = cateTrainTest.shape[1]

    precision = np.zeros((numTest, 1))
    recall = np.zeros((numTest, 1))

    topk = HammingRank[:k, :]

    for qid in range(numTest):
        retrieved = topk[:, qid]
        rel = cateTrainTest[retrieved, qid]
        retrieved_relevant_num = np.sum(rel)
        real_relevant_num = np.sum(cateTrainTest[:, qid])

        precision[qid] = retrieved_relevant_num/(k*1.0)
        recall[qid] = retrieved_relevant_num/(real_relevant_num*1.0)

    return precision.mean(), recall.mean()

if __name__ == '__main__':
    hashcode = np.array([[1,0,1,1,0],[0,1,0,1,0],[0,0,1,0,1],[1,0,0,1,0],[1,0,0,0,1],[0,1,0,1,0],[0,0,1,1,0],[0,1,0,1,0]])
    labels = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0],[0,1,0,0]])
    print 'labels:\n',labels
    hashcode[hashcode==0]=-1
    print hashcode

    hammingDist = 0.5*(-np.dot(hashcode,hashcode.transpose())+5)
    print 'hammingDist: \n',hammingDist
    HammingRank = np.argsort(hammingDist, axis=0)
    print 'Hamming Rank: \n',HammingRank

    sim_matrix = np.dot(labels,labels.transpose())
    print 'sim_matrix: \n',sim_matrix
    map = mAP(sim_matrix,HammingRank)
    print map


