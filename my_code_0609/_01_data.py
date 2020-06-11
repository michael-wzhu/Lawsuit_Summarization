import json
import re
import jieba
import pickle
import random
import numpy as np
import torch


input_path = "../sfzy_small/sfzy_small.json"
output_path = "/output/result.json"


def get_batch_index(num_sample, batch_size):
	num_batchs = np.ceil(num_sample / batch_size)
	i = 0
	indices = list(range(num_sample))
	random.shuffle(indices)

	while (i+1) * batch_size <= num_sample:
		yield indices[i*batch_size: min((i+1)*batch_size, num_sample)]
		i += 1


def get_batch_data(data, batch_index):
	return [data[i] for i in batch_index]


def process_batch_data(batch_data):
	summary = []
	labels = []
	sources = []
	sents = []
	sents_len = []
	words_len = []

	for d in batch_data:
		words_len.append([])
		summary.append(d[1])
		labels.append(d[4])
		sources.append(d[2])
		sents.append(d[3])
		sents_len.append(len(d[3]))

		for i, s in enumerate(d[3]):
			if len(s):
				words_len[-1].append(len(s))
			else:
				sents[-1].pop(i)
				labels[-1].pop(i)
				sources[-1].pop(i)

	batch_size = len(batch_data)

	with open("../input/dicts.pkl", "rb") as p:
		vocab, dict_w2i, dict_i2w = pickle.load(p)

	xs = []
	for i in range(batch_size):
		max_sent_len = max(words_len[i])
		x = np.zeros([sents_len[i], max_sent_len], dtype=np.int32)

		for j, sent in enumerate(sents[i]):
			for k, word in enumerate(sent):
				word_id = dict_w2i[word] if word in dict_w2i.keys() else dict_w2i["[UNK]"]
				x[j, k] = word_id
		xs.append(x)

	max_sent_num = max(sents_len)

	labels_ = np.zeros([batch_size, max_sent_num], dtype=np.float32)
	labels_mask = np.zeros([batch_size, max_sent_num], dtype=np.int32)
	for i, label_i in enumerate(labels):
		for j, _ in enumerate(label_i):
			labels_[i, j] = labels[i][j]
			labels_mask[i, j] = 1

	return xs, sources, summary, sents_len, words_len, labels_, labels_mask


def load_data(mode="train"):
	if mode == "train":
		with open('../input/train.pkl', 'rb') as p:
			data = pickle.load(p)
	else:
		with open('../input/test.pkl', 'rb') as p:
			data = pickle.load(p)
	with open("../input/dicts.pkl", "rb") as p:
		vocab, _, _ = pickle.load(p)

	return data, len(vocab)


def construct_dicts(words, vocab_size=30000, min_tf=1):
	dict_full = {}
	for w in words:
		if w in dict_full: dict_full[w] += 1
		else: dict_full[w] = 1

	vocab_full = sorted(dict_full.items(), key=lambda x: x[1], reverse=True)
	vocab = ["[PAD]", "[UNK]"]
	for v in vocab_full:
		if v[1] >= min_tf: vocab.append(v[0])
	vocab = vocab[:vocab_size]

	dict_i2w = dict(zip(list(range(len(vocab))), vocab))
	dict_w2i = dict(zip(vocab, list(range(len(vocab)))))

	with open("../input/dicts.pkl", "wb") as p:
		pickle.dump((vocab, dict_w2i, dict_i2w), p)


def processSourceText(text, all_words):
	# 逐个读取列表元素
	sents = []
	labels = []
	src = []

	for t in text:
		sent = t["sentence"].replace('\u3000', '').replace('\x20', '').replace('\xa0', '').replace(' ', '')
		src.append(sent)
		sent = re.sub('[0-9a-zA-Z]+', ' # ', sent)
		sents.append(sent)
		labels.append(t["label"])

	sents_cut = []
	for i in range(len(sents)):
		sent_cut = [word for word in jieba.cut(sents[i], cut_all=False) if word != ' ']
		sents_cut.append(sent_cut)
		all_words.extend(sent_cut)

	return src, sents_cut, labels


if __name__ == "__main__":
	with open(input_path, 'r', encoding="utf8") as f:
		processed_data = []
		all_words = []
		for line in f:
			data = json.loads(line)
			id = data.get("id")  # a string
			text = data.get("text")  # list of dicts
			summary = data.get("summary")  # a string

			sents, sents_cut, labels = processSourceText(text, all_words)

			processed_data.append([id, summary, sents, sents_cut, labels])

		random.shuffle(processed_data)
		train_size = int(len(processed_data) * 0.95)

	with open('../input/train.pkl', 'wb') as p:
		pickle.dump(processed_data[:train_size], p)
	with open('../input/test.pkl', 'wb') as p:
		pickle.dump(processed_data[train_size:], p)

	# with open('../input/train.pkl', 'rb') as p:
	# 	train_data = pickle.load(p)
	# with open('../input/test.pkl', 'rb') as p:
	# 	test_data = pickle.load(p)

	construct_dicts(all_words)