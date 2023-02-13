from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def entropy(data):

    if data.shape[0] == 0:
        return 0

    p0 = data[data['y']==0].shape[0] / data.shape[0]
    p1 = 1 - p0

    entropy = -p0*np.log2(p0+1e-9) - p1*np.log2(p1+1e-9)
    return entropy

def InfoGainRatio(data, var, splitVal):
    group1 = data.loc[data[var]<=splitVal]
    group2 = data.loc[data[var]>splitVal]

    # stopping criteria: entropy of candidate split is zero
    if group1.shape[0] == 0 or group2.shape[0] == 0:
        print("making leaf...")
        return 'make leaf!'
        # return entropy(data) - entropy(group1) - entropy(group2)

    w1 = group1.shape[0]/(group1.shape[0] + group2.shape[0])
    w2 = group2.shape[0]/(group1.shape[0] + group2.shape[0])

    info_gain = entropy(data) - (w1*entropy(group1) + w2*entropy(group2))

    intrinsic_info = - (w1*np.log2(w1) + w2*np.log2(w2))

    return info_gain/intrinsic_info

def DetermineCandidateSplits(data):
    candidates=[]
    df_sort_x1 = data.sort_values('x1')
    counter = 0
    for i in range(1, df_sort_x1.shape[0]):
        if df_sort_x1.iloc[i, 2] == df_sort_x1.iloc[i-1, 2]:
            continue
        else:
            if df_sort_x1.iloc[i,0] == 0.0:
                candidates.append(('x1', data[data['x1']==df_sort_x1.iloc[i,0]].index[counter]))
                counter += 1
            else:
                candidates.append(('x1', data[data['x1']==df_sort_x1.iloc[i,0]].index[0]))
    df_sort_x2 = data.sort_values('x2')
    for i in range(1, df_sort_x2.shape[0]):
        if df_sort_x2.iloc[i, 2] == df_sort_x2.iloc[i-1, 2]:
            continue
        else:
            candidates.append(('x2', data[data['x2']==df_sort_x2.iloc[i,1]].index[0]))

    return candidates

def MakeSplit(data, var, val):
    '''
    Given a data and a split conditions, do the split.
    variable: variable with which make the split.
    value: value of the variable to make the split.
    data: data to be splitted.
    '''
    data1 = data[data[var] <= val]
    data2 = data[(data[var] <= val) == False]

    return data1, data2

def MakePrediction(data):
    # predict 1 if there is no majority class in leaf
    if data[data['y']==1].shape[0] >= data[data['y']==0].shape[0]:
        prediction = 1
    else:
        prediction = 0
    return prediction

def MakeSubtree(data):
    data = data.reset_index(drop=True)

    # stopping criteria: empty node
    if data.shape[0] == 0:
        prediction = MakePrediction(data)
        return prediction

    # stopping criteria: empty node
    # if all(data['y']) or not any(data['y']):
    #     prediction = MakePrediction(data)
    #     return prediction

    total_splits = DetermineCandidateSplits(data)
    igr=[] 

    for i in range(len(total_splits)):
        if total_splits[i][0] == 'x1': 
            igr.append(InfoGainRatio(data, 'x1', data.iloc[total_splits[i][1],0]))
        else:
            igr.append(InfoGainRatio(data, 'x2', data.iloc[total_splits[i][1],1]))

    # stopping criteria: entropy of any candidate splits is zero
    for i in igr:
        if i=="make leaf!":
            prediction = MakePrediction(data)
            return prediction
    
    # stopping criteria: all splits have zero gain ratio
    if all(element == 0 for element in igr):
        prediction = MakePrediction(data)
        return prediction

    splitVar = total_splits[np.argmax(igr)][0]

    if splitVar == 'x1':
        splitVal = data.iloc[total_splits[np.argmax(igr)][1], 0]
    else:
        splitVal = data.iloc[total_splits[np.argmax(igr)][1], 1]

    left, right = MakeSplit(data, splitVar, splitVal) 

    splitType = "<="
    question = "{} {}  {}".format(splitVar, splitType, splitVal)

    subtree = {question: []}
    thenAnswer = MakeSubtree(left)
    elseAnswer = MakeSubtree(right)

    subtree[question].append(thenAnswer)
    subtree[question].append(elseAnswer)

    return subtree


# question 2.2

plt.plot(1,1, marker='o', color='red', label='0')
plt.plot(1,2, marker='o',color='green',label='1')
plt.plot(2,1, marker='o',color='green',label='1')
plt.plot(2,2, marker='o',color='red', label='0')
plt.legend()
plt.savefig('zerogainratio.pdf')


# question 2.3
# Druns = pd.read_csv('data/Druns.txt', delimiter=' ', names=['x1', 'x2', 'y'])
# splits = DetermineCandidateSplits(Druns)
# IGR=[]
# for i in range(len(splits)):
#     if splits[i][0] == 'x1': 
#         IGR.append(InfoGainRatio(Druns, 'x1', Druns.iloc[splits[i][1],0]))
#     else:
#         IGR.append(InfoGainRatio(Druns, 'x2', Druns.iloc[splits[i][1],1]))
# print(Druns, splits)

# for i in range(len(splits)):
#     if splits[i][0] == 'x1':
#         print("Splitting at index: ",splits[i][1], ", where", splits[i][0], "<=", Druns.iloc[splits[i][1],0], "yields information gain ratio of", IGR[i])
#     else:
#         print("Splitting at index: ",splits[i][1], ", where", splits[i][0], "<=", Druns.iloc[splits[i][1],1], "yields information gain ratio of", IGR[i])


# question 2.4
# d3leaves = pd.read_csv('data/D3leaves.txt', delimiter=' ', names=['x1', 'x2', 'y'])
# print(d3leaves)
# print(MakeSubtree(d3leaves))

# question 2.5 
# df = pd.read_csv('data/D1.txt', delimiter=' ', names=['x1', 'x2', 'y'])
# print(MakeSubtree(df))
# df = pd.read_csv('data/D2.txt', delimiter=' ', names=['x1', 'x2', 'y'])
# # print(MakeSubtree(df))

# question 2.6
# plt.scatter(df['x1'].loc[df['y']==0], df['x2'].loc[df['y']==0], label='y=0')
# plt.scatter(df['x1'].loc[df['y']==1], df['x2'].loc[df['y']==1], label='y=1')
# plt.legend()
# plt.title('D2.txt scatter')
# # plt.savefig('D2scatter.pdf')

# x = np.arange(20)/20
# print(x)
# y = [0.2]*20
# plt.plot(x,-x+1., color='red')

# plt.savefig('D2scatterDecisionBoundary.pdf')

# question 2.7
# df = pd.read_csv('data/Dbig.txt', delimiter=' ', names=['x1', 'x2', 'y'])
# df = df.sample(frac=1) # random permutation of original dataset

# train = df.iloc[:8192]
# test = df.iloc[8192:]

# D32 = train.iloc[:32]
# D128 = train.iloc[:128]
# D512 = train.iloc[:512]
# D2048 = train.iloc[:2048]
# D8192 = train

# D32_tree = MakeSubtree(D32)
# D128_tree = MakeSubtree(D128)
# D512_tree = MakeSubtree(D512)
# D2048_tree = MakeSubtree(D2048)
# D8192_tree = MakeSubtree(D8192)

# print(D32_tree,'\n' ,D128_tree, '\n',D512_tree, '\n',D2048_tree, '\n',D8192_tree)

# plt.semilogx([32, 128, 512, 2048, 8192], [1-0.3838495575221239, 1-0.7488938053097345, 1-0.6957964601769911, 1-0.8130530973451328, 1-0.6963495575221239 ])
# plt.xlabel('training set num_samples')
# plt.ylabel('test set error')
# plt.savefig('2.7.pdf')
# plt.show()

# plt.axhline(-0.709668)
# plt.vlines(0.901643, ymin=-0.709668, ymax=2, color='red')
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.title('$D_{128}$ tree decision boundary')
# plt.ylim(-2,2)
# plt.xlim(-2,2)
# plt.text(0,-1.5, '1')
# plt.text(-1,1, '0')
# plt.text(1.5,1, '1')
# plt.savefig('d128decisionboundary.pdf')
# plt.show()


# question 3: sklearn
# from sklearn.tree import DecisionTreeClassifier

# size = [D32, D128, D512, D2048, D8192]
# num_leaves = []
# errors = []
# for i in size:
#     clf  = DecisionTreeClassifier()
#     sklearn_tree = clf.fit(i.drop('y', axis=1), i['y'])
#     errors.append(1 - sklearn_tree.score(test.drop('y', axis=1), test['y']))
#     num_leaves.append(sklearn_tree.get_n_leaves())

# plt.plot(num_leaves, errors)
# plt.xlabel('num_leaves in tree')
# plt.ylabel('test set error')
# plt.savefig('sklearn_leaves_vs_error.pdf')
# plt.show()

