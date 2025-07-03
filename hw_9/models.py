from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple

import numpy as np

from preprocessing import TokenizedSentencePair


class BaseAligner(ABC):
    """
    Describes a public interface for word alignment models.
    """

    @abstractmethod
    def fit(self, parallel_corpus: List[TokenizedSentencePair]):
        """
        Estimate alignment model parameters from a collection of parallel sentences.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
        """
        pass

    @abstractmethod
    def align(self, sentences: List[TokenizedSentencePair]) -> List[List[Tuple[int, int]]]:
        """
        Given a list of tokenized sentences, predict alignments of source and target words.

        Args:
            sentences: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            alignments: list of alignments for each sentence pair, i.e. lists of tuples (source_pos, target_pos).
            Alignment positions in sentences start from 1.
        """
        pass


class DiceAligner(BaseAligner):
    def __init__(self, num_source_words: int, num_target_words: int, threshold=0.5):
        self.cooc = np.zeros((num_source_words, num_target_words), dtype=np.uint32)
        self.dice_scores = None
        self.threshold = threshold

    def fit(self, parallel_corpus):
        for sentence in parallel_corpus:
            # use np.unique, because for a pair of words we add 1 only once for each sentence
            for source_token in np.unique(sentence.source_tokens):
                for target_token in np.unique(sentence.target_tokens):
                    self.cooc[source_token, target_token] += 1
        self.dice_scores = (2 * self.cooc.astype(np.float32) /
                            (self.cooc.sum(0, keepdims=True) + self.cooc.sum(1, keepdims=True)))

    def align(self, sentences):
        result = []
        for sentence in sentences:
            alignment = []
            for (i, source_token), (j, target_token) in product(
                    enumerate(sentence.source_tokens, 1),
                    enumerate(sentence.target_tokens, 1)):
                if self.dice_scores[source_token, target_token] > self.threshold:
                    alignment.append((i, j))
            result.append(alignment)
        return result


class WordAligner(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters

    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        """
        Given a parallel corpus and current model parameters, get a posterior distribution over alignments for each
        sentence pair.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            posteriors: list of np.arrays with shape (src_len, target_len). posteriors[i][j][k] gives a posterior
            probability of target token k to be aligned to source token j in a sentence i.
        """
        posteriors = []
        for sentence_pair in parallel_corpus:
            teta = self.translation_probs[sentence_pair.source_tokens[:, None], sentence_pair.target_tokens]
            posteriors.append(teta / teta.sum(axis = 0))
            
        return posteriors
            
    def _compute_elbo(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]) -> float:
        """
        Compute evidence (incomplete likelihood) lower bound for a model given data and the posterior distribution
        over latent variables.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo: the value of evidence lower bound
        """
        elbo = 0
        for el in range(len(parallel_corpus)):
            n, m = [len(parallel_corpus[el].source_tokens), len(parallel_corpus[el].target_tokens)]
            teta = self.translation_probs[parallel_corpus[el].source_tokens[:, None], 
                                           parallel_corpus[el].target_tokens] 
            #elbo += (posteriors[el] * np.log((1/n)*teta)).sum() - (posteriors[el] * np.log(posteriors[el])).sum()
            elbo += (posteriors[el] * np.log(teta)).sum() - m*np.log(n) - (posteriors[el] * np.log(posteriors[el])).sum()
        return elbo
            

    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo:  the value of evidence lower bound after applying parameter updates
        """
        self.translation_probs = np.zeros((self.num_source_words, self.num_target_words), dtype=np.float32)
        denominators = np.zeros(self.num_source_words, dtype=np.float32)
        for el in range(len(parallel_corpus)):
            np.add.at(denominators, parallel_corpus[el].source_tokens, posteriors[el].sum(axis = 1))
            np.add.at(self.translation_probs, (parallel_corpus[el].source_tokens[:, None], 
            parallel_corpus[el].target_tokens), posteriors[el])
            
        self.translation_probs /= denominators.reshape(-1, 1)       
        elbo = self._compute_elbo(parallel_corpus, posteriors)
        return elbo
                                        

    def fit(self, parallel_corpus):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.
        Sorry for not sticking to my own interface ;)

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for i in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)
        return history

    def align(self, sentences):
        #posteriors = self._e_step(sentences)
        result = []
        for el in range(len(sentences)):
            targets_idx = np.arange(len(sentences[el].target_tokens)) + 1
            teta = self.translation_probs[sentences[el].source_tokens[:, None], 
                                           sentences[el].target_tokens]
            sources_idx = np.argmax(teta, axis = 0) + 1 
            result.append(sorted(list(zip(sources_idx, targets_idx)), key = lambda w: w[0]))
        return result


class WordPositionAligner(WordAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.alignment_probs = {}

    def _get_probs_for_lengths(self, src_length: int, tgt_length: int):
        """
        Given lengths of a source sentence and its translation, return the parameters of a "prior" distribution over
        alignment positions for these lengths. If these parameters are not initialized yet, first initialize
        them with a uniform distribution.

        Args:
            src_length: length of a source sentence
            tgt_length: length of a target sentence

        Returns:
            probs_for_lengths: np.array with shape (src_length, tgt_length)
        """
        pass

    def _e_step(self, parallel_corpus):
        pass

    def _compute_elbo(self, parallel_corpus, posteriors):
        pass

    def _m_step(self, parallel_corpus, posteriors):
        pass
