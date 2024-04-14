from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple

WORD = 0
TAG = 1


def prefixes(word: str, max_length: int = 5) -> List[str]:
    """
    Extracts prefixes from a word.

    Args:

word (str): The input word.
max_length (int): The maximum length of prefixes to extract. Defaults to 3.

    Returns:

List[str]: A list of prefixes extracted from the word."""
    return [word[:i] for i in range(1, min(max_length, len(word) + 1))]


def suffixes(word: str, max_length: int = 5) -> List[str]:
    """
    Extracts suffixes from a word.

    Args:

word (str): The input word.
max_length (int): The maximum length of suffixes to extract. Defaults to 3.

    Returns:

List[str]: A list of suffixes extracted from the word."""
    return [word[-i:] for i in range(1, min(max_length, len(word) + 1))]


class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        feature_dict_list = ["f100", "f101", "f102", "f103", "f104", "f105", "f106",
                             "f107", "f_digit", "f_capital", "f_alpha", "f_alnum", "f_title", "f_upper",
                             "f_length"]  # the
        # feature classes used in the code
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}
        '''
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        '''
        self.tags = set()  # a set of all the seen tags
        self.tags.add("~")
        self.tags.add("*")
        self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the test

    # f100
    def get_word_tag_pair_count(self, file_path) -> None:
        """
            Extract out of text all word/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-2]
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    self.tags.add(cur_tag)
                    self.tags_counts[cur_tag] += 1
                    self.words_count[cur_word] += 1
                    # f100
                    if (cur_word, cur_tag) not in self.feature_rep_dict["f100"]:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] += 1
                    # f101
                    for suffix in suffixes(cur_word):
                        if (suffix, cur_tag) not in self.feature_rep_dict["f101"]:
                            self.feature_rep_dict["f101"][(suffix, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f101"][(suffix, cur_tag)] += 1
                    # f102
                    for prefix in prefixes(cur_word):
                        if (prefix, cur_tag) not in self.feature_rep_dict["f102"]:
                            self.feature_rep_dict["f102"][(prefix, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f102"][(prefix, cur_tag)] += 1
                    # f103
                    if word_idx >= 2:
                        p_word, p_tag = split_words[word_idx - 1].split('_')
                        _, pp_tag = split_words[word_idx - 2].split('_')
                        if (pp_tag, p_tag, cur_tag) not in self.feature_rep_dict["f103"]:
                            self.feature_rep_dict["f103"][(pp_tag, p_tag, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f103"][(pp_tag, p_tag, cur_tag)] += 1
                    # f104
                    if word_idx >= 1:
                        p_word, p_tag = split_words[word_idx - 1].split('_')
                        if (p_tag, cur_tag) not in self.feature_rep_dict["f104"]:
                            self.feature_rep_dict["f104"][(p_tag, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f104"][(p_tag, cur_tag)] += 1
                    # f105
                    if cur_tag not in self.feature_rep_dict["f105"]:
                        self.feature_rep_dict["f105"][cur_tag] = 1
                    else:
                        self.feature_rep_dict["f105"][cur_tag] += 1
                    # f106
                    if word_idx >= 1:
                        p_word, _ = split_words[word_idx - 1].split('_')
                        if (p_word, cur_tag) not in self.feature_rep_dict["f106"]:
                            self.feature_rep_dict["f106"][(p_word, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f106"][(p_word, cur_tag)] += 1
                    # f107
                    if word_idx < len(split_words) - 1:
                        # print(line, split_words[word_idx])
                        n_word, _ = split_words[word_idx + 1    ].split('_')
                        if (n_word, cur_tag) not in self.feature_rep_dict["f107"]:
                            self.feature_rep_dict["f107"][(n_word, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f107"][(n_word, cur_tag)] += 1

                    # f_digit
                    if any(char.isdigit() for char in cur_word):
                        if (cur_word, cur_tag) not in self.feature_rep_dict["f_digit"]:
                            self.feature_rep_dict["f_digit"][(cur_word, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f_digit"][(cur_word, cur_tag)] += 1

                    # f_capital
                    if not cur_word.islower():  # true if all letters are lower
                        if (cur_word, cur_tag) not in self.feature_rep_dict["f_capital"]:
                            self.feature_rep_dict["f_capital"][(cur_word, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f_capital"][(cur_word, cur_tag)] += 1

                    ## adding feats

                    # f_alpha
                    if cur_word.isalpha():  # true if all chars are (a-z)
                        if (cur_word, cur_tag) not in self.feature_rep_dict["f_alpha"]:
                            self.feature_rep_dict["f_alpha"][(cur_word, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f_alpha"][(cur_word, cur_tag)] += 1

                    # f_alnum
                    if cur_word.isalnum():  # true if all chars are (a-z and numbers)
                        if (cur_word, cur_tag) not in self.feature_rep_dict["f_alnum"]:
                            self.feature_rep_dict["f_alnum"][(cur_word, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f_alnum"][(cur_word, cur_tag)] += 1

                    # f_title
                    if cur_word.istitle():  # true if first letter of all words (one word) is capital
                        if (cur_word, cur_tag) not in self.feature_rep_dict["f_title"]:
                            self.feature_rep_dict["f_title"][(cur_word, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f_title"][(cur_word, cur_tag)] += 1

                    # f_upper
                    if cur_word.isupper():  # true if all letters are capital
                        if (cur_word, cur_tag) not in self.feature_rep_dict["f_upper"]:
                            self.feature_rep_dict["f_upper"][(cur_word, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f_upper"][(cur_word, cur_tag)] += 1

                    # f_length
                    if (len(cur_word), cur_tag) not in self.feature_rep_dict["f_length"]:
                        self.feature_rep_dict["f_length"][(len(cur_word), cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f_length"][(len(cur_word), cur_tag)] += 1

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                # history is tupple of  the last last word , the last word and therie tags and the next word without
                # a tag

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0])

                    self.histories.append(history)


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.feature_to_idx = {
            "f100": OrderedDict(), "f101": OrderedDict(), "f102": OrderedDict(), "f103": OrderedDict(),
            "f104": OrderedDict(), "f105": OrderedDict(), "f106": OrderedDict(), "f107": OrderedDict(),
            "f_digit": OrderedDict(), "f_capital": OrderedDict(), "f_alpha": OrderedDict(), "f_title": OrderedDict(),
            "f_upper": OrderedDict(), "f_length": OrderedDict(), "f_alnum": OrderedDict()
        }

        self.feature_thresh = {}

        for feature, thresh in zip(self.feature_to_idx.keys(), threshold):
            self.feature_thresh[feature] = thresh

        self.histories_matrix = OrderedDict()
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_feature_to_idx(self):
        return self.feature_to_idx

    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_id
        """
        for feat_class in self.feature_statistics.feature_rep_dict:
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.feature_thresh[feat_class]:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        """
        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []
        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in feat_to_vec(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for r, y_tag in enumerate(self.feature_statistics.tags):
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
                self.histories_features[demi_hist] = []
                for c in feat_to_vec(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                            shape=(len(self.feature_statistics.tags) * len(
                                                self.feature_statistics.histories), self.n_total_features),
                                            dtype=bool)
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(
                self.feature_statistics.histories), self.n_total_features), dtype=bool)


def feat_to_vec(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]]) \
        -> List[int]:
    """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word} # TODO: change this
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all features that are relevant to the given history
    """
    c_word = history[0]
    c_tag = history[1]
    p_word = history[2]
    p_tag = history[3]
    pp_word = history[4]
    pp_tag = history[5]
    n_word = history[6]
    features = []
    # n_word = history[6]

    # f100
    if (c_word, c_tag) in dict_of_dicts["f100"]:
        features.append(dict_of_dicts["f100"][(c_word, c_tag)])
    # f101
    for suffix in suffixes(c_word):
        if (suffix, c_tag) in dict_of_dicts["f101"]:
            features.append(dict_of_dicts["f101"][(suffix, c_tag)])
    # f102
    for prefix in prefixes(c_word):
        if (prefix, c_tag) in dict_of_dicts["f102"]:
            features.append(dict_of_dicts["f102"][(prefix, c_tag)])
    # f103
    if (pp_tag, p_tag, c_tag) in dict_of_dicts["f103"]:
        features.append(dict_of_dicts["f103"][(pp_tag, p_tag, c_tag)])
    # f104
    if (p_tag, c_tag) in dict_of_dicts["f104"]:
        features.append(dict_of_dicts["f104"][(p_tag, c_tag)])
    # f105
    if c_tag in dict_of_dicts["f105"]:
        features.append(dict_of_dicts["f105"][c_tag])
    # f106
    if (p_word, c_tag) in dict_of_dicts["f106"]:
        features.append(dict_of_dicts["f106"][(p_word, c_tag)])
    # f107
    if (n_word, c_tag) in dict_of_dicts["f107"]:
        features.append(dict_of_dicts["f107"][(n_word, c_tag)])
    # f_digit
    if (c_word, c_tag) in dict_of_dicts["f_digit"]:
        features.append(dict_of_dicts["f_digit"][(c_word, c_tag)])
    # f_capital
    if (c_word, c_tag) in dict_of_dicts["f_capital"]:
        features.append(dict_of_dicts["f_capital"][(c_word, c_tag)])
    # f_alpha
    if (c_word, c_tag) in dict_of_dicts["f_alpha"]:
        features.append(dict_of_dicts["f_alpha"][(c_word, c_tag)])

    # f_alnum
    if (c_word, c_tag) in dict_of_dicts["f_alnum"]:
        features.append(dict_of_dicts["f_alnum"][(c_word, c_tag)])

    # f_title
    if (c_word, c_tag) in dict_of_dicts["f_title"]:
        features.append(dict_of_dicts["f_title"][(c_word, c_tag)])

    # f_upper
    if (c_word, c_tag) in dict_of_dicts["f_upper"]:
        features.append(dict_of_dicts["f_upper"][(c_word, c_tag)])

    # f_length
    if (len(c_word), c_tag) in dict_of_dicts["f_length"]:
        features.append(dict_of_dicts["f_length"][(len(c_word), c_tag)])
    return features




def preprocess_train(train_path, threshold):
    # Statistics
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    # feature2id
    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()
    print(feature2id.n_total_features)

    for dict_key in feature2id.feature_to_idx:
        print(dict_key, len(feature2id.feature_to_idx[dict_key]))
    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences

