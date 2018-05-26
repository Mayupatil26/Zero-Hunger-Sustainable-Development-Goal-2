#!/usr/bin/env python
# -*- coding: utf-8 -*- 

# Reference for the code: http://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/

from bitarray import bitarray
import mmh3
import math
import sys

#Class file of the bloom filter
#Bloom Filter data structure to find the false positive of the words
#mmh3 technique is used for the hash function 
 
class BloomFilter(object): 
    def __init__(self,number_of_items,probability_false_positive):
        self.size = self.size_of_hash_function(number_of_items,probability_false_positive) 
        self.number_of_hf = self.get_number_of_hf(self.size,number_of_items)
        self.bucket = bitarray(self.size)
        self.probability_false_positive = probability_false_positive
        self.bucket.setall(0)

#Checking if the item is present in the bloom filter 
    def checking_item_bf(self, word):
        for i in range(self.number_of_hf):
            a = mmh3.hash(word,i)%self.size
            if self.bucket[a] == False:
                return False
        return True

#Adding item to the bloomfilter
    def adding_item_bf(self, word):
        a = []
        for i in range(self.number_of_hf):
            b=mmh3.hash(word,i)%self.size
            a.append(b)
            self.bucket[b] = True
#Class methods implemented 
#Size of the hash function 
    @classmethod
    def size_of_hash_function(self,i,j):
        size_of_hash_function = -(i * math.log(j))/(math.log(2)**2)
        return int(size_of_hash_function)
#Number of hash functions
    @classmethod
    def get_number_of_hf(self, i, j):
        number = (i/j) * math.log(2)
        return int(number)