from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    with open(filename, 'r', encoding = 'utf-8') as f:
        xml = f.read().replace('&', '&amp;')
    sentences = []
    alignments = []
    
    root = ET.fromstring(xml)
    
    for child in root:
        english = child.find('english').text
        czech = child.find('czech').text
        sure = child.find('sure').text
        possible = child.find('possible').text
    
        sentences.append(SentencePair(english.split(), czech.split()))
        if sure != None:
            sure = sure.split()
            sure_new = []
            for el in sure:
                sure_new.append((int(el.split('-')[0]), int(el.split('-')[1])))
        else:
            sure_new = []
        if possible != None:
            possible = possible.split()
            possible_new = []
            for el in possible:
                possible_new.append((int(el.split('-')[0]), int(el.split('-')[1])))
        else:
            possible_new = []
        alignments.append(LabeledAlignment(sure_new, possible_new))

    
    return sentences, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    count_source = {}
    count_target = {}
    for el in sentence_pairs:
        for word in el.source:
            if word in count_source:
                count_source[word] += 1
            else:
                count_source[word] = 1
        for word in el.target:
            if word in count_target:
                count_target[word] += 1
            else:
                count_target[word] = 1
    if freq_cutoff != None:
        count_source = {k: v for k, v in sorted(count_source.items(), key = lambda x: x[1], reverse = True)[:freq_cutoff]}
        count_target = {k: v for k, v in sorted(count_target.items(), key = lambda x: x[1], reverse = True)[:freq_cutoff]}
    final_source = {list(count_source.keys())[i]: i for i in range(len(count_source.keys()))}  
    final_target = {list(count_target.keys())[i]: i for i in range(len(count_target.keys()))} 
    return final_source, final_target


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tknzd_sents = []
    for el in sentence_pairs:
        
        source_list = []
        target_list = []
        for word in el.source:
            if word in source_dict:
                source_list.append(source_dict[word])
        for word in el.target:
            if word in target_dict:
                target_list.append(target_dict[word])
                
        if (source_list == []) or (target_list == []):
            pass
        else:
            tknzd_sents.append(TokenizedSentencePair(np.array(source_list), np.array(target_list)))
            
    return tknzd_sents
