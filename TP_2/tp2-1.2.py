#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# read input data
# supermarket =
# print(supermarket.head())

# convert categorical values into one-hot vectors
def one_hot_encode(x):
    if x == '?':
        return False
    elif x == 't':
        return True
    else:
        raise ValueError
supermarket_one_hot = supermarket.drop(['total'], axis=1).applymap(one_hot_encode)
def one_hot_encode_total(x):
    if x == 'high':
        return False
    elif x == 'low':
        return True
    else:
        raise ValueError
supermarket_one_hot = supermarket.drop(['total'], axis=1).applymap(one_hot_encode)
for val in supermarket['total'].unique():
    print(val)
    supermarket_one_hot['total_'+str(val)] = supermarket['total'].apply(lambda x: x==val)
print(supermarket_one_hot.head())


# option to show all itemsets
pd.set_option('display.max_colwidth',None)

# frequent_itemsets = apriori(...)

# rules=association_rules(frequent_itemsets, ...

# select rules with more than 2 antecedents
# rules.loc[map(lambda x: len(x)>2,rules['antecedents'])]


