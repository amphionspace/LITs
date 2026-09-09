#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   front_utils.py
@Time    :   2024/08/22 22:10:29
@Author  :   Bo Jin 
@Version :   1.0
@Contact :   jinbo5650@gmail.com
@Brief   :   放前端数据处理工具
'''

import os



def merge_word(word_list:list, span_list:list):
    word_list_label = []
    for i in range(len(word_list)):
        w = word_list[i]
        word_list_label += [i] * len(w)
    # print(word_list_label)
    for j in range(len(span_list)):
        span = span_list[j]
        word_list_label[span[0]:span[-1]] = [-j - 1] * (span[-1] - span[0])
    word_txt = "".join(word_list)
    assert len(word_txt) == len(word_list_label)
    new_word_list = [word_txt[0]]
    front_label = word_list_label[0]
    for k in range(1, len(word_list_label)):
        if word_list_label[k] != front_label:
            new_word_list.append(word_txt[k])
        else:
            new_word_list[-1] += word_txt[k]
        front_label = word_list_label[k]
    return new_word_list


if __name__ == "__main__":
    word_list = ["您好", "姐", "姐", "我", "想", "姐", "姐"]
    span_list = [(2,4), (6,8)]
    print(merge_word(word_list, span_list))