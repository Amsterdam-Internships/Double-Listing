import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib
import random
import re

def scoring_unsupervised(df,baseline =False):
    if baseline == False:

        # Adapted from https://github.com/wbsg-uni-mannheim/UnsupervisedBootAL on 2022-16-04
        # Only using the scores
        other_columns  = df.drop(['ListingId_1','ListingId_2','ids'], axis=1)
        #Including nan values to count the densities
        other_columns = other_columns.replace(-1.0,np.nan)
        #Column densities
        column_weights = []
        for c in other_columns:
            nan_values = other_columns[c].isna().sum()
            ratio = float(nan_values)/float(len(other_columns[c]))
            column_weights.append(1.0-ratio)
        #Create normalized values
        weighted_columns = other_columns*column_weights
        other_columns_sum = weighted_columns.sum(axis=1, skipna=True)
        other_columns_mean = other_columns_sum/len(other_columns.columns)
        other_columns_mean = np.interp(other_columns_mean, (other_columns_mean.min(), other_columns_mean.max()), (0, +1))

def scoring_unsupervised(df,baseline =False):
    if baseline == False:

        # Adapted from https://github.com/wbsg-uni-mannheim/UnsupervisedBootAL on 2022-16-04
        # Only using the scores
        other_columns  = df.drop(['ListingId_1','ListingId_2','ids'], axis=1)
        #Including nan values to count the densities
        other_columns = other_columns.replace(-1.0,np.nan)
        #Column densities
        column_weights = []
        for c in other_columns:
            nan_values = other_columns[c].isna().sum()
            ratio = float(nan_values)/float(len(other_columns[c]))
            column_weights.append(1.0-ratio)
        #Create normalized values
        weighted_columns = other_columns*column_weights
        other_columns_sum = weighted_columns.sum(axis=1, skipna=True)
        other_columns_mean = other_columns_sum/len(other_columns.columns)
        other_columns_mean = np.interp(other_columns_mean, (other_columns_mean.min(), other_columns_mean.max()), (0, +1))

        sorted_dataset = list(zip(df['ids'], other_columns_mean, np.arange(df['ids'].size)))
        #random.Random(0).shuffle(sorted_dataset)
        sorted_dataset.sort(key = lambda t: t[1])
        #sorted_dataset = map(lambda x:x[1],sorted_dataset)
        return sorted_dataset
    else:
        # Adapted from https://github.com/wbsg-uni-mannheim/UnsupervisedBootAL on 2022-16-04
        #Only using the scores
        other_columns  = df.drop(['source_id','target_id','ids'], axis=1)
        #Including nan values to count the densities
        other_columns = other_columns.replace(-1.0,np.nan)
        #Column densities
        column_weights = []
        for c in other_columns:
            nan_values = other_columns[c].isna().sum()
            ratio = float(nan_values)/float(len(other_columns[c]))
            column_weights.append(1.0-ratio)
        #Create normalized values
        weighted_columns = other_columns*column_weights
        other_columns_sum = weighted_columns.sum(axis=1, skipna=True)
        other_columns_mean = other_columns_sum/len(other_columns.columns)
        other_columns_mean = np.interp(other_columns_mean, (other_columns_mean.min(), other_columns_mean.max()), (0, +1))
        sorted_dataset = list(zip(df['ids'], other_columns_mean, np.arange(df['ids'].size)))
        #random.Random(0).shuffle(sorted_dataset)
        sorted_dataset.sort(key = lambda t: t[1])
        #sorted_dataset = map(lambda x:x[1],sorted_dataset)
        return sorted_dataset

def elbow_threshold(sorted_dataset):
    #Retrieved from https://github.com/wbsg-uni-mannheim/UnsupervisedBootAL on 2022-16-04
    sim_scores = [sim[1] for sim in sorted_dataset]
    nPoints = len(sim_scores)
    allCoord = np.vstack((range(nPoints), sim_scores)).T
    firstPoint = allCoord[0]
    # get vector between first and last point - this is the line
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    plt.plot(range(nPoints),sim_scores)
    plt.axline((0,0), (nPoints,1), color='black',linestyle = '--',lw=1)
    plt.plot(idxOfBestPoint,sim_scores[idxOfBestPoint],'ro')
    plt.show()
    print("Knee of the curve is at index =",idxOfBestPoint)
    print("Knee value =", sim_scores[idxOfBestPoint])
    return sim_scores[idxOfBestPoint],idxOfBestPoint

def unsupervised_labels(sorted_dataset,threshold_value):
    # Adapted from https://github.com/wbsg-uni-mannheim/UnsupervisedBootAL on 2022-16-04
    # Get scores and ids
    sim_scores = [sim[1] for sim in sorted_dataset]
    sim_ids = [sim[0] for sim in sorted_dataset]
    # Find the threshold index on sim_scores
    threshold_index = sim_scores.index(threshold_value)
    # Divide between the matches and non matches with threshold
    sim_ids_nonmatches = sim_ids[:threshold_index]
    sim_ids_matches= sim_ids[-(len(sorted_dataset)-threshold_index):]
    sim_score_nonmatches = sim_scores[:threshold_index]
    sim_score_matches= sim_scores[-(len(sorted_dataset)-threshold_index):]
    # Calculate the condifence weights
    weights_nonmatches = abs(sim_scores[:threshold_index]-threshold_value)/(threshold_value-min(sim_scores))
    weights_matches = abs(sim_scores[-(len(sorted_dataset)-threshold_index):]-threshold_value)/(max(sim_scores)-threshold_value)
    # Join all
    matches_score_weight = list(zip(sim_ids_matches,sim_score_matches,weights_matches,np.ones(len(sim_ids_matches))))
    nonmatches_score_weight = list(zip(sim_ids_nonmatches,sim_score_nonmatches,weights_nonmatches,np.zeros(len(sim_ids_nonmatches))))
    return matches_score_weight, nonmatches_score_weight