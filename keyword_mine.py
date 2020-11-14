#!/usr/bin/python3
import argparse
from lime.lime_text import LimeTextExplainer
from simple_explainer import SimpleExplainer
import numpy as np
from tokenization import tokenize
import jnius_config

jnius_config.add_classpath('/Users/weisu.yxd/IdeaProjects/nlp-web/target/nlp-web-1.0.jar')
from jnius import autoclass

ArrayList = autoclass('java.util.ArrayList')
Normalizer = autoclass('com.alibaba.security.util.Normalizer')
Politics = autoclass('com.alibaba.security.model2.Politics')

normalizer = Normalizer.getInstance()
politics = Politics('/Users/weisu.yxd/IdeaProjects/nlp-web/data/')
class_names = ['normal', 'political']


def predict_proba(txts):
    inputs = ArrayList()
    for txt in txts:
        inputs.add(txt)
    scores = politics.predictProb(inputs)
    proba = np.empty([len(txts), 2], dtype=np.float32)
    for idx, score in enumerate(scores):
        p = 1. - score
        proba[idx, ] = [p, score]
    return proba


explainer = LimeTextExplainer(class_names=class_names, split_expression=tokenize, verbose=False)
simple_explainer = SimpleExplainer(split_expression=tokenize, ngram=1)


def simple_explain(text, normalize=True, threshold=1e-3):
    if normalize:
        text = normalizer.evaluate(text)
    words = simple_explainer.explain_instance(text, predict_proba, num_features=10)
    if words is None:
        return None
    words = list(filter(lambda x: x[1] > threshold, words))
    return words


def explain(text, filter_negative=True, normalize=True):
    if normalize:
        text = normalizer.evaluate(text)
    if not text:
        return None
    # tokens = tokenize(text)
    # length = len(tokens)
    # num_words = 10 if length > 10 else length
    # if length <= 16:
    #     n_samples = 1024
    # elif length <= 32:
    #     n_samples = 2048
    # else:
    #     n_samples = 3096
    num_words = 10
    n_samples = 1024
    exp = explainer.explain_instance(text, predict_proba, num_features=num_words, num_samples=n_samples)
    words = list(filter(lambda x: x[1] > 0, exp.as_list())) if filter_negative else exp.as_list()
    return words


def extract_keywords(infile, outfile, explain_fn):
    inf = open(infile)
    out_f = open(outfile, 'w')
    count = 0
    for line in inf:
        text = normalizer.evaluate(line.rstrip())
        if not text:
            continue
        count += 1
        if count % 10 == 0:
            print(f'{count} records processed.')
        start = 0
        idx = text.rfind(' ', 0, 280)
        inputs = ArrayList()
        while idx > 0:
            sub = text[start: idx]
            start = idx + 1
            inputs.clear()
            inputs.add(sub)
            scores = politics.predictProb(inputs)
            if scores.get(0) < 0.5:
                idx = text.rfind(' ', start, start + 280)
                continue
            if len(sub) <= 8:
                out_f.write(sub + ':1\001' + sub + '\n')
                continue
            keywords = explain_fn(sub, normalize=False)
            result = '\t'.join([w[0] + ':%.5f' % w[1] for w in keywords])
            out_f.write(result + '\001' + sub + '\n')
            if len(text) - idx < 10:
                break
            idx = text.rfind(' ', start, start + 280)
        if start == 0:
            inputs.clear()
            text = text[:300]
            inputs.add(text)
            scores = politics.predictProb(inputs)
            if scores.get(0) > 0.5:
                keywords = explain_fn(text, normalize=False)
                result = '\t'.join([w[0] + ':%.5f' % w[1] for w in keywords])
                out_f.write(result + '\001' + text + '\n')
    inf.close()
    out_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Implementation of the explanation algorithm.')
    parser.add_argument('--input', type=str, help='input file of examples to be explained')
    parser.add_argument('--output', type=str, help='output file of explained result')
    parser.add_argument('--explainer', type=str, help='output file of explained result')
    args = vars(parser.parse_args())
    for k, v in args.items():
        print('{} = {}'.format(k, v))

    explain_fn = explain if args['explainer'] == 'lime' else simple_explain
    if None not in (args['input'], args['output']):
        print("prepare to extract keywords ...")
        extract_keywords(args['input'], args['output'], explain_fn)

    # sent = "瘟神又要忙活收中共刽子手了"
    # norm_sent = normalizer.evaluate(sent)
    # tokens = tokenize(norm_sent)
    # print(len(tokens), tokens)
    # result = explain(sent)
    # print(result)
    # result = simple_explain(sent, threshold=1e-4)
    # print(result)
