from lime.lime_text import IndexedString
import numpy as np


class SimpleExplainer(object):
    def __init__(self, split_expression=r'\W+', ngram=1, threshold=0.5, mask_string=None):
        self.split_expression = split_expression
        self.mask_string = mask_string
        self.threshold = threshold
        assert ngram >= 1
        self.ngram = ngram

    def __data_labels(self, indexed_string, classifier_fn):
        """generate new sample, removing one word each time"""
        doc_size = indexed_string.num_words()
        inverse_data = [indexed_string.raw_string()]
        for n in range(self.ngram):
            ngram = n + 1
            for i in range(0, doc_size - n):
                inactive = [i + x for x in range(ngram)]
                inverse_data.append(indexed_string.inverse_removing(inactive))
        labels = classifier_fn(inverse_data)
        return labels

    def explain_instance(self, text_instance, classifier_fn, num_features, label=1):
        indexed_string = IndexedString(text_instance, bow=True,
                                       split_expression=self.split_expression,
                                       mask_string=self.mask_string)
        labels = self.__data_labels(indexed_string, classifier_fn)
        base = labels[0][label]
        if base < self.threshold:
            print("Not Hit")
            return None
        num_samples = len(labels) - 1

        num_words = indexed_string.num_words()
        distance = np.zeros(num_samples, dtype=np.float32)
        for i in range(0, num_samples - 1):
            distance[i] = labels[i + 1][label] - base  # 越小越好，为了排序
            if i < num_words:
                continue
            word_ids = self.get_word_ids(num_words, i)
            for idx in word_ids:
                if distance[i] > distance[idx]:
                    distance[i] = 0  # n-gram 去除后模型预测得分反而大于只去除 1 个term的情况，无效
                    break

        idx = np.argsort(distance)
        result = []
        for i in range(min(num_features, num_samples)):
            start = idx[i]
            snippet = self.get_text_snippet(indexed_string, start)
            score = -distance[start]
            result.append((snippet, score))
        return result

    def get_word_ids(self, num_words, start):
        ngram = 1
        upper = num_words
        offset = 0
        while upper <= start:
            length = num_words - ngram
            upper += length
            offset += length + 1
            ngram += 1
        start -= offset
        word_ids = [start + offset for offset in range(ngram)]
        return word_ids

    def get_text_snippet(self, indexed_string, start):
        num_words = indexed_string.num_words()
        word_ids = self.get_word_ids(num_words, start)
        words = [indexed_string.word(idx) for idx in word_ids]
        return ''.join(words)
