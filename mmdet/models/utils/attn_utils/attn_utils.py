# -*- coding: utf-8 -*-
# !@time: 2020/12/30 下午12:31
# !@author: superMC @email: 18758266469@163.com
# !@fileName: attn_utils.py


def compute_reflectiveField(kernel_size_list):
    percentage_list = [1]
    reflective_field = 1
    for kernel_size in kernel_size_list:
        new_list = [1 / kernel_size for i in range(kernel_size, 0, -1)]
        reflective_field += kernel_size - 1
        new_cache_list = [0] * reflective_field
        for i in range(len(percentage_list)):
            for j in range(len(new_list)):
                new_cache_list[i + j] += percentage_list[i] * new_list[j]
        percentage_list = new_cache_list
    return reflective_field, percentage_list


def compute_repeat_ratio(kernel_size_list):
    reflective_field, percentage_list = compute_reflectiveField(kernel_size_list)
    ratio_list = [0] * reflective_field
    for i in range(reflective_field):
        for j in range(i, reflective_field):
            ratio_list[i] += min(percentage_list[j - i], percentage_list[j])
    return ratio_list


if __name__ == '__main__':
    # reflective_field, cache_list = compute_reflectiveField([3, 3, 3])
    # for cache in cache_list:
    #     print(cache * 3 ** 3)
    ratio_list = compute_repeat_ratio([3, 3, 3])
    print(ratio_list)
