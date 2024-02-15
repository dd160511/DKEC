import numpy as np
import sqlite3
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics
import torch.nn.functional as F

def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro

# def eva(y_true, y_pred, epoch=0):
#     acc, f1 = cluster_acc(y_true, y_pred)
#     nmi = nmi_score(y_true, y_pred)
#     ari = ari_score(y_true, y_pred)
#     print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
#             ', f1 {:.4f}'.format(f1))

def eva(y_true, y_pred, epoch=0, bacth=0, datasetname=None):
    flag=False
    acc, f1 = cluster_acc(y_true, y_pred)
    # nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    print('dataname:{0},batch:{1}'.format(datasetname,bacth),epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1))

    if epoch == 'pae':
        return

    con = sqlite3.connect("result.db")
    cur = con.cursor()

    hasdata = cur.execute("select * from sdcn where batch = ? and datasetname = ?",
                            [bacth, datasetname]).fetchone()
    if hasdata == None:
        cur.execute("insert into sdcn(batch,epoch,acc,nmi,ari,f1,datasetname) values(?,?,?,?,?,?,?)",
                    [bacth, epoch, acc, nmi, ari, f1, datasetname])
        con.commit()

    oldresult = cur.execute("select * from sdcn where nmi<? and batch = ? and datasetname = ?",
                            [nmi, bacth, datasetname]).fetchone()
    if oldresult != None:
        cur.execute("update sdcn set acc=?,nmi=?,ari=?,f1=?,epoch=? where batch=? and datasetname=?",
                    [acc, nmi, ari, f1, epoch, bacth, datasetname])
        flag=True
        con.commit()

    cur.close()
    con.close()
    return nmi,nmi,nmi,flag
# def eva(y_true, y_pred, pdatas,epoch=0, bacth=0, datasetname=None):
#     flag=False
#     acc, f1 = cluster_acc(y_true, y_pred)
#     # nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
#     nmi = nmi_score(y_true, y_pred)
#     ari = ari_score(y_true, y_pred)
#     print('dataname:{0},batch:{1}'.format(datasetname,bacth),epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
#           ', f1 {:.4f}'.format(f1))
#
#     if epoch == 'pae':
#         return
#
#     con = sqlite3.connect("result.db")
#     cur = con.cursor()
#
#     hasdata = cur.execute("select * from sdcn where batch = ? and datasetname = ?",
#                             [bacth, datasetname]).fetchone()
#     if hasdata == None:
#         cur.execute("insert into sdcn(batch,epoch,acc,nmi,ari,f1,datasetname) values(?,?,?,?,?,?,?)",
#                     [bacth, epoch, acc, nmi, ari, f1, datasetname])
#         con.commit()
#
#     oldresult = cur.execute("select * from sdcn where nmi<? and batch = ? and datasetname = ?",
#                             [nmi, bacth, datasetname]).fetchone()
#     if oldresult != None:
#         cur.execute("update sdcn set acc=?,nmi=?,ari=?,f1=?,epoch=? where batch=? and datasetname=?",
#                     [acc, nmi, ari, f1, epoch, bacth, datasetname])
#         for key in pdatas:
#             label = F.softmax(pdatas[key], dim=1).data.cpu().numpy().argmax(1)
#             #label = y_true
#             filename = "{0}_{1}_{2}".format(datasetname,bacth, key)
#             np.savetxt(r"./hdatas/{0}.txt".format(filename),pdatas[key].data.cpu().numpy(),fmt='%.8f')
#             np.savetxt(r"./hdatas/{0}_label.txt".format(filename),label,fmt='%d')
#         flag=True
#         con.commit()
#
#     cur.close()
#     con.close()
#     return flag