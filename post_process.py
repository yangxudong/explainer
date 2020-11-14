#!/usr/bin/python3
import argparse
from tokenization import tokenize
from lime.lime_text import IndexedString
import itertools
from tqdm import tqdm


def min_distance(indexed_string, word_id1, word_id2):
    if word_id1 == word_id2:
        return 0
    if abs(word_id2 - word_id1) == 1:
        return 1
    pos1 = indexed_string.positions[word_id1]
    pos2 = indexed_string.positions[word_id2]
    min_dist = indexed_string.num_words()
    for p1 in pos1:
        for p2 in pos2:
            dist = abs(p1 - p2)
            if dist <= 1:
                return dist
            if dist < min_dist:
                min_dist = dist
    return min_dist


def merge(text_instance, words):
    """words: a dict map from word to weight, and the order matters"""
    indexed_string = IndexedString(text_instance, bow=True, split_expression=tokenize)
    num_words = indexed_string.num_words()
    word2index = {}
    indices = []
    for i in range(num_words):
        word = indexed_string.word(i)
        if word in words:
            word2index[word] = i
            indices.append(i)

    candidate = []
    concat_words = set()
    max_weight = 0
    for w1, w2 in itertools.combinations(words, 2):
        if w1 in concat_words or w2 in concat_words:
            continue
        # if w1.isalpha() and len(w1) == 1:
        #     continue
        # if w2.isalpha() and len(w2) == 1:
        #     continue
        if w1 not in word2index:
            print(f"warning: {w1} do not exist in index")
            continue
        if w2 not in word2index:
            print(f"warning: {w2} do not exist in index")
            continue
        id1 = word2index[w1]
        id2 = word2index[w2]
        dist = min_distance(indexed_string, id1, id2)
        if dist == 1:
            weight = words[w1] + words[w2]
            sep = ' ' if w1.isalpha() and w2.isalpha() else ''
            phrase = sep.join([w1, w2]) if id1 < id2 else sep.join([w2, w1])
            if weight > max_weight:
                max_weight = weight
                candidate.append(phrase)
                concat_words.add(w1)
                concat_words.add(w2)
            elif weight > 0.1 * max_weight:
                candidate.append(phrase)
                concat_words.add(w1)
                concat_words.add(w2)

    threshold = max_weight * 0.1
    for word, weight in words.items():
        if word in concat_words:
            continue
        if weight < threshold:
            continue
        candidate.append(word)

    candidate.sort()
    return '&'.join(candidate)


def process(infile, outfile):
    out_f = open(outfile, 'w')
    inf = open(infile)
    for line in tqdm(inf):
        elements = line.rstrip().split('\001')
        sentence = elements[1]
        if len(sentence) <= 8:
            out_f.write(sentence + '\001' + sentence + '\001' + elements[0] + '\n')
            continue
        word_info = elements[0].split()
        words = {info.split(':')[0]: float(info.split(':')[1]) for info in word_info}
        rules = merge(elements[1], words)
        out_f.write(rules + '\001' + sentence + '\001' + elements[0] + '\n')
    inf.close()
    out_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post process of the explanation algorithm.')
    parser.add_argument('--input', type=str, help='input file of examples to be explained')
    parser.add_argument('--output', type=str, help='output file of explained result')
    args = vars(parser.parse_args())

    if None not in (args['input'], args['output']):
        print("prepare to generate rules ...")
        process(args['input'], args['output'])

    # text = '斩妖除魔中共洗脑害人民不让人民信佛神江妖迫害法轮功魔鬼邪恶残暴凶一亿法徒讲真相二十一年遭迫害迷中世人被魔骗仇恨反对法轮功讥笑辱骂大法徒法徒救人心不动大法弟子慈悲劝忍辱负重传真相可怜世人中毒深相信中共丧了命中共罪恶已满盈瘟神领旨灭邪灵大劫降临淘汰人亲共信共染毒病神要铲除红魔鬼伴随中共难活命天涯海角无处躲不弃中共必丧命各国官民选未来真善忍好度众生'
    # tokens = tokenize(text)
    # s = ''.join(tokens)
    # if text == s:
    #     print("yes")
    # else:
    #     print("no")
    # print(tokens)
