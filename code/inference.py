import numpy as np
import scipy.sparse as sp
from code.preprocessing import read_test, feat_to_vec
from tqdm import tqdm
from typing import Dict, Tuple


class Viterbi:
    def __init__(self, pre_trained_weights, feature2id, beam_width):

        self.next_possible_u = None
        self.Pi = None
        self.pre_trained_weights = pre_trained_weights
        self.feature2id = feature2id
        self.tags = np.array(list(feature2id.feature_statistics.tags))
        self.n_tags = len(self.tags)
        self.curr_feature_mat = sp.lil_matrix((self.n_tags, self.feature2id.n_total_features), dtype=int)
        self.curr_sentence = None
        self.curr_Q = None
        self.curr_tags = None
        self.backpointer = None
        self.curr_history = tuple()
        # Create a dictionary to map indices to tags
        self.index_to_tag = {i: tag for i, tag in enumerate(self.tags)}
        # Create a dictionary to map tags to indices
        self.tag_to_index = {tag: i for i, tag in enumerate(self.tags)}
        self.q_dict: Dict[Tuple, np.ndarray] = {}
        self.beam_width = beam_width

    def get_q(self, key: tuple) -> np.ndarray:
        """
        return a softmax vector for a give history i.e q(v|t,u,w,i) for all possible values
        if not already calculated then store in ou dict of tuples
        """
        curr_w, prev_w, p_tag, pp_w, pp_tag, next_w = key
        probs = []
        if key not in self.q_dict:
            for i in range(self.n_tags):
                hist = (curr_w, self.index_to_tag[i], prev_w, p_tag, pp_w, pp_tag, next_w)
                dict_of_dicts = self.feature2id.feature_to_idx
                numerator = np.exp(np.sum(self.pre_trained_weights[feat_to_vec(hist, dict_of_dicts)]))
                probs.append(numerator)
            self.q_dict[key] = np.array(probs) / sum(probs)
        return self.q_dict[key]

    def calc_Q(self, curr_index, prev_tag_index, possible_pp_tags):
        """
        creating Q, row by row, this mat is dim tags*tags and Q(i,j) is probability of going from pptag_i to
        prev_tag(fixed) to curr tag i
        """

        prev_tag = self.index_to_tag[prev_tag_index]
        Q = np.zeros(shape=(self.n_tags, self.n_tags))
        for i in possible_pp_tags:
            pp_tag = self.index_to_tag[i]
            key = (self.curr_sentence[curr_index], self.curr_sentence[curr_index - 1], prev_tag,
                   self.curr_sentence[curr_index - 2], pp_tag, self.curr_sentence[curr_index + 1])
            Q[i, :] = self.get_q(key)
        self.curr_Q = Q

    def backtrack(self, override=True):
        num_words = len(self.curr_sentence)
        # The last word is tagged with the tag that has the maximum viterbi probability
        self.curr_tags = [None] * num_words
        tags_idxs = [None] * num_words
        tags_idxs[-2], tags_idxs[-1] = np.unravel_index(np.argmax(self.Pi[-2], axis=None),
                                                        self.Pi[-2].shape)  # what does this return?
        self.curr_tags[-1], self.curr_tags[-2] = self.index_to_tag[tags_idxs[-1]], self.index_to_tag[tags_idxs[-2]]
        # Then we go backwards from the last word to the first
        for i in range(num_words - 3, 0, -1):
            # The tag of the i-th word is the backpointer of the (i+1)-th word and its tag
            tags_idxs[i] = self.backpointer[i + 2 - 1, tags_idxs[i + 1], tags_idxs[i + 2]]  # -1 for the offset
            self.curr_tags[i] = self.index_to_tag[tags_idxs[i]]
        self.curr_tags.append('~')
        self.curr_tags = self.curr_tags[1:]
        # slice here to fix problem
        if override:
            symbols_to_match = ["~", "*", ",", ".", ":", "#", '"', "`", "$"]
            # Iterate over each word in the sentence
            for i, word in enumerate(self.curr_sentence):
                # Check if the word matches any of the specified symbols
                if word in symbols_to_match:
                    # Set the corresponding index in curr_tags to the same symbol
                    self.curr_tags[i] = word

    def predict(self, sentence) -> list:
        self.curr_sentence = sentence
        num_words = len(self.curr_sentence)
        self.Pi = np.zeros((num_words, self.n_tags, self.n_tags))
        BOS = self.tag_to_index['*']
        self.Pi[1, BOS, BOS] = 1  # -> init pi(0,*,*)
        self.backpointer = np.zeros(self.Pi.shape, dtype=int)
        for k in range(2, len(self.curr_sentence) - 1):  # not sure the window length
            non_zero_indices = np.argwhere(self.Pi[k - 1])
            indexed_values = [(idx[0], idx[1], self.Pi[k - 1, idx[0], idx[1]]) for idx in non_zero_indices]
            sorted_indices = sorted(indexed_values, key=lambda x: x[2], reverse=True)
            possible_p_p_tags = set([idx[0] for idx in sorted_indices[:self.beam_width]])
            possible_p_tags = set([idx[1] for idx in sorted_indices[:self.beam_width]])
            for p_tag_index in possible_p_tags:  # then only iterate through prev possible tags
                self.calc_Q(k, p_tag_index, possible_p_p_tags)
                col = self.Pi[k - 1, :, p_tag_index].reshape(-1, 1)
                pi_k_u_v = self.curr_Q * col  # -> no dimension change, element wise column multiplication
                self.Pi[k, p_tag_index, :], self.backpointer[k, p_tag_index, :] = np.max(pi_k_u_v,
                                                                                         axis=0), np.argmax(
                    pi_k_u_v, axis=0)
            #
            #
            # print(k)
            # for p_tag_index in range(self.n_tags):
            #     if Beam_Search:
            #         non_zero_indices = np.argwhere(self.Pi[k - 1])
            #         indexed_values = [(idx[0], idx[1], self.Pi[k - 1, idx[0], idx[1]]) for idx in non_zero_indices]
            #         sorted_indices = sorted(indexed_values, key=lambda x: x[2], reverse=True)
            #         possible_p_p_tags = [idx[0] for idx in sorted_indices[:beam_size]]
            #     else:
            #         possible_p_p_tags = np.nonzero(self.Pi[k - 1, :, p_tag_index])[0]
            #     if len(possible_p_p_tags) > 0:
            #         self.calc_Q(k, p_tag_index, possible_p_p_tags)
            #         col = self.Pi[k - 1, :, p_tag_index].reshape(-1, 1)
            #         pi_k_u_v = self.curr_Q * col  # -> no dimension change, element wise column multiplication
            #         self.Pi[k, p_tag_index, :], self.backpointer[k, p_tag_index, :] = np.max(pi_k_u_v,
            #                                                                                  axis=0), np.argmax(
            #             pi_k_u_v, axis=0)

        self.backtrack()
        return self.curr_tags[1:]


def memm_viterbi(sentence, pre_trained_weights, feature2id, beam_width):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """

    viterbi = Viterbi(pre_trained_weights, feature2id, beam_width)
    return viterbi.predict(sentence)


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path, beam_width=2):
    tagged = "test" in test_path
    # tagged = True  # TODO: comment!!!!!!!!!!
    test = read_test(test_path, tagged=tagged)  # test -> [list of tuples like this:([list of words],[list of tags])]

    output_file = open(predictions_path, "w")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]  # sen -> ([list of words],[list of tags])
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id, beam_width)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
