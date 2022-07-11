import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re
from time import perf_counter

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])
#print (parents_df.head())

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]
#print (df.head())

def transform_name(product_name):
    #IMPLEMENT
    product_name = product_name.lower()
    pattern = re.compile(r"[^a-z0-9]")
    product_name = re.sub(pattern, " ",product_name)
    product_words = product_name.split()
    product_words = [stemmer.stem(word) for word in product_words]
    product_name = " ".join(product_words)
    return product_name

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.

df["query"] = df["query"].apply(transform_name)

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.

# q = __import__("functools").partial(__import__("os")._exit, 0)  # FIXME
# __import__("IPython").embed()  # FIXME  

#category_counts_dict = df["category"].value_counts().to_dict()

def get_leaves(family_dict,counts_dict):
    children = set(family_dict.keys())
    parents = set(family_dict.values())
    leaves = children - parents
    return leaves

def collapse_family(category_query_dict, family_dict,prune_dict,threshold):
    leaves = get_leaves(family_dict,category_query_dict)
    pass_count = 0
    collapse_count = 0
    unqueried_count = 0
    for category in leaves:
        parent = family_dict.pop(category)
        try:
            queries = category_query_dict.pop(category)
        except KeyError:
            unqueried_count += 1
            continue
        if len(queries) >= threshold:
            prune_dict[category] = queries
            pass_count += 1
        else:
            parent_queries = category_query_dict.get(parent,[])
            parent_queries += queries
            category_query_dict[parent] = parent_queries
            collapse_count += 1

    print (f"Processed {len(leaves)} categories. {pass_count} passed, {collapse_count} collapsed, Unqueried {unqueried_count}  ")
    print (f"counts_dict size {len(category_query_dict)}, prune_dict size {len(prune_dict)}, family_dict_size {len(family_dict)}\n")
    return category_query_dict, family_dict,prune_dict

def prune_categories(category_query_dict, family_dict,threshold = 100):
    print ("Starting pruning")
    prune_dict = {}
    print (f"counts_dict size {len(category_query_dict)}. prune_dict size {len(prune_dict)}\n")
    while len(category_query_dict) > 1:
        category_query_dict, family_dict,prune_dict = collapse_family(category_query_dict, family_dict,prune_dict,threshold)
    prune_dict.update(category_query_dict)
    return prune_dict



start = perf_counter()
category_query_dict = df.groupby('category')['query'].apply(list).to_dict()
parents_dict = dict(zip(parents_df["category"], parents_df["parent"]))
prune_dict = prune_categories(category_query_dict, parents_dict,threshold = 1000)
category_query_pairs = []
for category,queries in prune_dict.items():
    for query in queries:
        pair = (category,query)
        category_query_pairs.append(pair)

df = pd.DataFrame(category_query_pairs, columns =['category', 'query'])

end = perf_counter()
time_taken = end - start
print (f"{time_taken=}")
# q = __import__("functools").partial(__import__("os")._exit, 0)  # FIXME
# __import__("IPython").embed()  # FIXME  

# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
