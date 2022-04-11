import jaro
from strsimpy.levenshtein import Levenshtein
import math
from recordlinkage.algorithms.distance import _haversine_distance
from recordlinkage.algorithms.numeric import _linear_sim
from recordlinkage.utils import fillna as _fillna
from functools import partial



def get_jaro_winker(str1,str2):
    #Names similarities
    #If one of strings is nan it will be a float 
    if type(str1) == float or type(str2) == float:
        #if there is one nan retrun -1, tho only float can be checked like this math.isnan(str1) so use try
        try:
            if math.isnan(str1):
                return -1.0
        except TypeError:
            if math.isnan(str2):
                return -1.0
    else:
        return jaro.jaro_winkler_metric(str1,str2)


def get_jaccard_sim(str1, str2):
    # Retreived from https://github.com/wbsg-uni-mannheim/UnsupervisedBootAL/blob/master/code/similarityutils.py  on 2022/01/11 at 11:43
    #If one of strings is nan it will be a float 
    if type(str1) == float or type(str2) == float:
        #if there is one nan retrun -1, tho only float can be checked like this math.isnan(str1) so use try
        try:
            if math.isnan(str1):
                return -1.0
        except TypeError:
            if math.isnan(str2):
                return -1.0
    else:
        a = set(str1.split())
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / float(len(a) + len(b) - len(c))


def get_relaxed_jaccard_sim(str1, str2):
    # Retreived from https://github.com/wbsg-uni-mannheim/UnsupervisedBootAL/blob/master/code/similarityutils.py  on 2022/01/11 at 11:43
    #If one of strings is nan it will be a float 
    if type(str1) == float or type(str2) == float:
        #if there is one nan retrun -1, tho only float can be checked like this math.isnan(str1) so use try
        try:
            if math.isnan(str1):
                return -1.0
        except TypeError:
            if math.isnan(str2):
                return -1.0
    else: 
        a = set(str1.split())
        b = set(str2.split())
        c = []
        for a_ in a:
            for b_ in b:
                if get_levenshtein_sim(a_, b_) > 0.7:
                    c.append(a_)

        intersection = len(c)
        min_length = min(len(a), len(b))
        if intersection > min_length:
            intersection = min_length
        return float(intersection) / float(len(a) + len(b) - intersection)


def get_containment_sim(str1, str2):
    # Retreived from https://github.com/wbsg-uni-mannheim/UnsupervisedBootAL/blob/master/code/similarityutils.py  on 2022/01/11 at 11:43
    #If one of strings is nan it will be a float 
    if type(str1) == float or type(str2) == float:
        #if there is one nan retrun -1, tho only float can be checked like this math.isnan(str1) so use try
        try:
            if math.isnan(str1):
                return -1.0
        except TypeError:
            if math.isnan(str2):
                return -1.0
    else:
        a = set(str1.split())
        b = set(str2.split())
        c = a.intersection(b)
        if len(a) == 0 or len(b) == 0:
            return -1.0
        else:
            return float(len(c)) / float(min(len(a), len(b)))


def get_levenshtein_sim(str1, str2):

    # Retreived from https://github.com/wbsg-uni-mannheim/UnsupervisedBootAL/blob/master/code/similarityutils.py  on 2022/01/11 at 11:43
 
    levenshtein = Levenshtein()
    #If one of strings is nan it will be a float 
    if type(str1) == float or type(str2) == float:
        #if there is one nan retrun -1, tho only float can be checked like this math.isnan(str1) so use try
        try:
            if math.isnan(str1):
                return -1.0
        except TypeError:
            if math.isnan(str2):
                return -1.0
    else:
        max_length = max(len(str1), len(str2))
        return 1.0 - levenshtein.distance(str1, str2) / max_length

def get_overlap_sim(str1, str2):
    # Retreived from https://github.com/wbsg-uni-mannheim/UnsupervisedBootAL/blob/master/code/similarityutils.py  on 2022/01/11 at 11:43
    #If one of strings is nan it will be a float 
    if type(str1) == float or type(str2) == float:
        #if there is one nan retrun -1, tho only float can be checked like this math.isnan(str1) so use try
        try:
            if math.isnan(str1):
                return -1.0
        except TypeError:
            if math.isnan(str2):
                return -1.0
    elif str1 == str2:
        return 1.0
    else:
        return 0.0

def get_binary(num_1,num_2):
    if type(num_1) == float or type(num_2) == float:
        if math.isnan(num_1) or math.isnan(num_2):
            return -1.0
    if num_1==num_2:
        return 1.0
    else:
        return 0.0

def get_numerical(num_1,num_2):
    #From book
    if type(num_1) == float or type(num_2) == float:
        if math.isnan(num_1) or math.isnan(num_2):
            return -1.0
    
    if num_1==num_2:
        return 1.0
    else:
        pc = (abs(num_1-num_2)/max(num_1,num_2))*100
        if pc < 33:
            return 1- (pc/33)
        else:
            return 0



def get_geo(lat1, lng1, lat2, lng2):
    #retrived from https://github.com/J535D165/recordlinkage/blob/master/recordlinkage/compare.py on 2022/01/11 at 11:43
    offset = 0.0
    scale = 1.0
    origin = 0.0
    

    if type(lat1) == float or type(lat2) == float or type(lng1) == float or type(lng2) == float:
        if math.isnan(lat1) or math.isnan(lng1) or math.isnan(lat2) or math.isnan(lng2):
            return -1.0

        else:
            d = _haversine_distance(lat1, lng1, lat2, lng2)


    
            num_sim_alg = partial(_linear_sim, d,scale,offset,origin)


            c = num_sim_alg()


            return c
    else:
        return 'LAT or LNG in worng format'
