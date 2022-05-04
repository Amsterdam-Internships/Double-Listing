import jaro
from strsimpy.levenshtein import Levenshtein
import math
from recordlinkage.algorithms.distance import _haversine_distance
from recordlinkage.algorithms.numeric import _linear_sim
from recordlinkage.utils import fillna as _fillna
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision import transforms
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from PIL import Image
import requests




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

def cos_similarity(embeddings_1, embeddings_2,norm =False):
    #Calculate the cosine similarity wither normalized or not
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # If there is a nan value return -1
    if type(embeddings_1) == float or type(embeddings_2) == float:
        return -1
    else:
        #Normalized for LaBSE     
        if norm ==True:
            if float(embeddings_1.sum()) != 0.0 and  float(embeddings_2.sum()) != 0:
                normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
                normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
                return float(cos(normalized_embeddings_1, normalized_embeddings_2))
            else:
                return -1
        else:
            if float(embeddings_1.sum()) != 0.0 and  float(embeddings_2.sum()) != 0:
                return float(cos(embeddings_1,embeddings_2))
            else:
                return -1

def transform_torch(list):
    if type(list) != float :
        #Tranforms str list with array which was converted wrongly '[1.2332,....,-0.323]'
        return torch.Tensor([[np.float(x) for x in list[1:-1].split(',')]])
    else:
        return np.float('nan')

def clean_tfidf(string):
    #Cleaning strings for tf-idf analysis
    snowball_eng = SnowballStemmer(language='english')
    snowball_nl = SnowballStemmer(language='dutch')

    stopword_list = stopwords.words('dutch') +  stopwords.words('english')
    if type(string) != float:
        token_words=word_tokenize(string)
        stem_sentence=[]
        for word in token_words:
            if word not in stopword_list:
                word = snowball_eng.stem(word)
                word = snowball_nl.stem(word)
                stem_sentence.append(word)
                stem_sentence.append(" ")
        return "".join(stem_sentence)
    else:
        # If np.nan return '' so there is not an error
        return ""

def retreive_img(url):
    #Retrieve the image from link into a numpy array
    try:
        if type(url) != float:
            im = Image.open(requests.get(url, stream=True).raw)
            im = im.convert("RGB")
            convert_tensor = transforms.ToTensor()
            im = convert_tensor(im)
            im = fn.resize(im, size=[224, 224])
            im= np.array(im)
            im = im.transpose(1,2,0)
            im = np.expand_dims(im, axis=0)
            return im
        else:
            im_total_arrays = np.zeros((224,224,3))
            im_total_arrays = np.expand_dims(im_total_arrays, axis=0)
            return im_total_arrays
    except Image.UnidentifiedImageError:
        im_total_arrays = np.zeros((224,224,3))
        im_total_arrays = np.expand_dims(im_total_arrays, axis=0)
        return im_total_arrays