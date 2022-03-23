# /*
#  * @Author: Yue.Fan 
#  * @Date: 2022-03-23 11:35:21 
#  * @Last Modified by:   Yue.Fan 
#  * @Last Modified time: 2022-03-23 11:35:21 
#  */
from __future__ import absolute_import, division, print_function

import collections
import json
import logging
import math
import numpy as np
from io import open
from tqdm import tqdm
import itertools

import re
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  whitespace_tokenize)


logger = logging.getLogger(__name__)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 is_yes=None,
                 is_no=None,
                 labels=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.is_yes = is_yes
        self.is_no = is_no
        self.labels = labels

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 paragraph_len,
                 label_id=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 unk_mask=None,
                 yes_mask=None,
                 no_mask=None,
                 answer_mask=None,
                 answer_num=None
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.paragraph_len = paragraph_len
        self.label_id = label_id
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.unk_mask = unk_mask
        self.yes_mask = yes_mask
        self.no_mask = no_mask
        self.answer_mask = answer_mask
        self.answer_num = answer_num


def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    change_ques_num = 0
    for entry in tqdm(input_data, desc='convert examples:'):
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]

            doc_tokens = []
            char_to_word_offset = []

            for c in paragraph_text:
                doc_tokens.append(c)
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                if question_text != qa['question']:
                    change_ques_num += 1
                start_position = None
                end_position = None
                orig_answer_text = None
                # 多答案：
                start_positions = []
                end_positions = []
                orig_answer_texts = []

                is_impossible = False
                is_yes = False
                is_no = False
                labels = ['O'] * len(doc_tokens)
                if is_training:
                    if version_2_with_negative:
                        if qa['is_impossible'] == 'false':
                            is_impossible = False
                        else:
                            is_impossible = True
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        continue

                    if not is_impossible:
                        all_answers = qa["answers"]
                        if len(all_answers) == 0:
                            answers = []
                        else:
                            answers = all_answers[0]
                        if type(answers) == dict:
                            answers = [answers]

                        for answer in answers:
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = "".join(doc_tokens[start_position:(end_position + 1)])

                            cleaned_answer_text = " ".join(
                                whitespace_tokenize(orig_answer_text))


                            if actual_text.find(cleaned_answer_text) == -1:

                                if cleaned_answer_text == 'YES':
                                    is_yes = True
                                    orig_answer_text = 'YES'
                                    start_position = -1
                                    end_position = -1

                                    labels = ['O'] * len(doc_tokens)

                                    start_positions.append(start_position)
                                    end_positions.append(end_position)
                                    orig_answer_texts.append(orig_answer_text)
                                elif cleaned_answer_text == 'NO':
                                    is_no = True
                                    start_position = -1
                                    end_position = -1
                                    orig_answer_text = 'NO'

                                    labels = ['O'] * len(doc_tokens)

                                    start_positions.append(start_position)
                                    end_positions.append(end_position)
                                    orig_answer_texts.append(orig_answer_text)
                                else:
                                    logger.warning("Could not find answer: '%s' vs. '%s'",
                                                   actual_text, cleaned_answer_text)
                                    continue
                            else:
                                start_positions.append(start_position)
                                end_positions.append(end_position)
                                orig_answer_texts.append(orig_answer_text)
                                start_index = answer['answer_start']
                                end_index = start_index + len(answer['text'])
                                labels[start_index: end_index] = ['I'] * (len(answer['text']))
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                        start_positions.append(start_position)
                        end_positions.append(end_position)
                        orig_answer_texts.append(orig_answer_text)
                        labels = ['O'] * len(doc_tokens)


                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_texts,
                    start_position=start_positions,
                    end_position=end_positions,
                    is_impossible=is_impossible,
                    is_yes=is_yes,
                    is_no=is_no,
                    labels=labels)
                examples.append(example)

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training, max_n_answers=1):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    unk_tokens = {}

    label_map = {'O': 0, 'I': 1}

    convert_token_list = {
        '“': '"', "”": '"', '…': '...', '﹤': '<', '﹥': '>', '‘': "'", '’': "'",
        '﹪': '%', 'Ⅹ': 'x', '―': '-', '—': '-', '﹟': '#', '㈠': '一'
    }
    example_index = 0
    for example in tqdm(examples, desc='convert_examples_to_features:'):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        labellist = example.labels

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            if "[UNK]" in sub_tokens:
                if token in unk_tokens:
                    unk_tokens[token] += 1
                else:
                    unk_tokens[token] = 1

            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        # multi answers
        tok_start_positions = []
        tok_end_positions = []

        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1

            tok_start_positions.append(-1)
            tok_end_positions.append(-1)

        if is_training and not example.is_impossible:
            for orig_answer_text, start_position, end_position in zip(example.orig_answer_text, example.start_position, example.end_position):

                tok_start_position = orig_to_tok_index[start_position]
                if end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    orig_answer_text)

                tok_start_positions.append(tok_start_position)
                tok_end_positions.append(tok_end_position)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

            while start_offset < len(all_doc_tokens) and all_doc_tokens[start_offset - 1] != "," and all_doc_tokens[start_offset - 1] != "。":   # 滑窗 保留完整一句话 此处是英文的，
                start_offset += 1

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            doc_span_labels = []
            label_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            label_ids.append(label_map["O"])

            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
                label_ids.append(label_map["O"])
            tokens.append("[SEP]")
            segment_ids.append(0)
            label_ids.append(label_map["O"])

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                doc_span_labels.append(labellist[split_token_index])
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
                label_ids.append(label_map[doc_span_labels[i]])
            paragraph_len = doc_span.length

            tokens.append("[SEP]")
            segment_ids.append(1)
            label_ids.append(label_map["O"])

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(label_map["O"])

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = example.is_impossible

            start_position = None
            end_position = None
            # multi answers
            start_positions = []
            end_positions = []

            if is_training and not example.is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1

                for tok_start_position, tok_end_position in zip(tok_start_positions, tok_end_positions):  # 多个答案 计算各自起始位置
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = max_seq_length
                        end_position = max_seq_length
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

                    start_positions.append(start_position)
                    end_positions.append(end_position)

            unk_mask, yes_mask, no_mask = [0], [0], [0]
            if is_training and example.is_impossible:   # no answer
                start_position = max_seq_length
                end_position = max_seq_length
                unk_mask = [1]

                if start_positions:
                    start_positions.clear()
                    end_positions.clear()

                    start_positions.append(start_position)
                    end_positions.append(end_position)
                else:
                    start_positions.append(start_position)
                    end_positions.append(end_position)

            elif is_training and example.is_yes:   # YES
                start_position = max_seq_length+1
                end_position = max_seq_length+1
                yes_mask = [1]

                if start_positions:
                    start_positions.clear()
                    end_positions.clear()

                    start_positions.append(start_position)
                    end_positions.append(end_position)
                else:
                    start_positions.append(start_position)
                    end_positions.append(end_position)

            elif is_training and example.is_no:   # NO
                start_position = max_seq_length+2
                end_position = max_seq_length+2
                no_mask = [1]

                if start_positions:
                    start_positions.clear()
                    end_positions.clear()

                    start_positions.append(start_position)
                    end_positions.append(end_position)
                else:
                    start_positions.append(start_position)
                    end_positions.append(end_position)

            # 如果答案列表长度 大于设置的阈值最大答案数量 则随机选择
            if len(start_positions) > max_n_answers:
                idxs = np.random.choice(len(start_positions), max_n_answers, replace=False)
                st = []
                en = []
                for idx in idxs:
                    st.append(start_positions[idx])
                    en.append(end_positions[idx])
                start_positions = st
                end_positions = en

            answer_num = len(start_positions) if not example.is_impossible else 0

            answer_mask = [1 for _ in range(len(start_positions))]
            for _ in range(max_n_answers - len(start_positions)):
                start_positions.append(0)
                end_positions.append(0)
                answer_mask.append(0)



            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    paragraph_len=paragraph_len,
                    start_position=start_positions,
                    end_position=end_positions,
                    is_impossible=example.is_impossible,
                    unk_mask=unk_mask,
                    yes_mask=yes_mask,
                    no_mask=no_mask,
                    answer_mask=answer_mask,
                    answer_num=answer_num,
                    label_id=label_ids
                ))
            unique_id += 1
        example_index += 1
    if is_training:
        with open("unk_tokens_clean", "w", encoding="utf-8") as fh:
            for key, value in unk_tokens.items():
                fh.write(key+" " + str(value)+"\n")

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def group_with_index(l):   # 返回连续片段 索引
    i = 0
    res = []
    for k, vs in itertools.groupby(l):
        c = sum(1 for _ in vs)
        # print (k, c, i)
        res.append([k, c, i])
        i += c
    return res

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    id2label = {0: 'O', 1: 'I'}
    span_xulie = []
    all_nbest_new = []

    num_span_duo = 0
    list_span_duo = []

    example_index_to_features = collections.defaultdict(list)
    # 将每个样例的不同片段加入到对应的list中， 一个example_index对应若干个unique_id
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    # 每个unique_id的答案
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = []
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    id2fakeanswer = {}
    for (example_index, example) in enumerate(all_examples):
        # 获得该样本所有片段
        features = example_index_to_features[example_index]

        IO_spans = []

        # 该样本的答案
        prelim_predictions = []
        fake_answer = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        answer_num = 1

        score_yes = 1000000
        min_yes_feature_index = 0  # the paragraph slice with min null score
        yes_start_logit = 0  # the start logit at the slice with min null score
        yes_end_logit = 0  # the end logit at the slice with min null score

        score_no = 1000000
        min_no_feature_index = 0  # the paragraph slice with min null score
        no_start_logit = 0  # the start logit at the slice with min null score
        no_end_logit = 0  # the end logit at the slice with min null score

        answer_num_ = 1

        for (feature_index, feature) in enumerate(features):
            # 对于某个片段，计算得分
            result = unique_id_to_result[feature.unique_id]

            IO_logits_list = result.IO_logits
            IO_final = []
            for i in  IO_logits_list:
                if i == 0:
                    IO_final.append(id2label[0])
                else:
                    IO_final.append(id2label[1])
            res = group_with_index(IO_final)    # 返回的是 (字母， 连续数量，开始位置)
            for r in res:
                if r[0] == 'I' and r[1] > 1:
                    start = r[2]
                    end = r[2] + r[1] - 1

                    if start not in feature.token_to_orig_map:
                        continue
                    if end not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start, False):
                        continue

                    tok_tokens = feature.tokens[start:(end + 1)]
                    orig_doc_start = feature.token_to_orig_map[start]
                    orig_doc_end = feature.token_to_orig_map[end]
                    orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                    tok_text = "".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = "".join(tok_text.split())
                    orig_text = "".join(orig_tokens)

                    final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)

                    IO_spans.append(final_text)

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)

            answer_nums = result.answer_num
            answer_num = answer_nums.index(max(answer_nums))

            if answer_num > 1:
                answer_num_ = answer_num

            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.unk_logits[0]*2
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.unk_logits[0]
                    null_end_logit = result.unk_logits[0]

                feature_yes_score = result.yes_logits[0] + result.yes_logits[0]
                if feature_yes_score < score_yes:
                    score_yes = feature_yes_score
                    min_yes_feature_index = feature_index
                    yes_start_logit = result.yes_logits[0]
                    yes_end_logit = result.yes_logits[0]

                feature_no_score = result.no_logits[0] + result.no_logits[0]
                if feature_no_score < score_no:
                    score_no = feature_no_score
                    min_no_feature_index = feature_index
                    no_start_logit = result.no_logits[0]
                    no_end_logit = result.no_logits[0]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
                    if start_index>0:
                        fake_answer.append(_PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        IO_spans_ = list(set(IO_spans))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=512,
                    end_index=512,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_yes_feature_index,
                    start_index=513,
                    end_index=513,
                    start_logit=yes_start_logit,
                    end_logit=yes_end_logit))
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_no_feature_index,
                    start_index=514,
                    end_index=514,
                    start_logit=no_start_logit,
                    end_logit=no_end_logit))
        # 排序
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)


        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "orig_doc_start", "orig_doc_end"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index < 512:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = "".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = "".join(tok_text.split())
                orig_text = "".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)

                overlap = False  # 有空改成算f1   计算非重叠span   如果有重叠，则跳过
                for item in nbest:
                    if item.text == 'YES' or item.text == 'NO' or len(item.text) == 0:   # 如果候选的是上述三类，则不参与计算非重叠子片段
                        continue
                    else:
                        if (orig_doc_start <= item.orig_doc_start and item.orig_doc_start <= orig_doc_end) or (
                                orig_doc_start <= item.orig_doc_end and item.orig_doc_end <= orig_doc_end):
                            overlap = True
                        elif item.text.find(final_text) != -1 or final_text.find(item.text) != -1:
                            overlap = True
                if overlap:
                    continue

                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit,
                        orig_doc_start=orig_doc_start,
                        orig_doc_end=orig_doc_end
                    ))
            elif pred.start_index == 512:
                final_text = ""
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit,
                        orig_doc_start=-1,
                        orig_doc_end=-1
                    ))
            elif pred.start_index == 513:
                final_text = "YES"
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit,
                        orig_doc_start=-1,
                        orig_doc_end=-1
                    ))
            else:
                final_text = "NO"
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit,
                        orig_doc_start=-1,
                        orig_doc_end=-1
                    ))

        assert len(nbest) >= 1

        total_scores = []
        # best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        
        # 选答案
        if answer_num_ > 1:
            num_span_duo += 1
            list_span_duo.append(example.qas_id)
            result_span = []
            for span in IO_spans_:
                if len(span) > 1:
                    result_span.append(span)
            all_predictions.append({"id": example.qas_id, "answer": result_span})
        else:
            all_predictions.append({"id": example.qas_id, "answer": [nbest_json[0]["text"]]})

        span_xulie.append({"id": example.qas_id, "IO_span": IO_spans_})

        all_nbest_json[example.qas_id] = nbest_json

    # print(num_span_duo)
    # print(list_span_duo)

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")
    
    # with open('nbest_new.json', "w") as writer:
    #     writer.write(json.dumps(all_nbest_new, indent=4, ensure_ascii=False) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(span_xulie, indent=4, ensure_ascii=False) + "\n")

    return all_predictions


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs