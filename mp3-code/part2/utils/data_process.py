# 将一个浮点数列表圆整为小数点后三位，输出一个txt来保存，同时将分隔符换为&

import sys
import os
import re
import math


# 将一个浮点数列表圆整为小数点后三位，输出一个txt来保存，同时将分隔符换为&
# def round_float_list(float_list, output_file_name):
#     with open(output_file_name, 'w') as output_file:
#         for float_num in float_list:
#             output_file.write(str(round(float_num, 3)) + '&')
#         output_file.write('\n')
#
#
# list_1 = [0.0, 0.8641975308641976, 0.0, 1.0, 0.9545454545454546, 0.8799999999999999, 0.7341772151898733,
#           0.9444444444444444, 0.7692307692307693, 0.9473684210526316, 0.946236559139785, 0.8571428571428571,
#           0.891566265060241, 0.6599999999999999]
# output_file_name = 'list_1.txt'
#
# round_float_list(list_1, output_file_name)


# company 0.0011152130441470544
# based 0.00042301184433164127
# business 0.00038455622211967387
# founded 0.00034610059990770653
# records 0.00030764497769573913
# bergen 0.00030764497769573913
# record 0.00030764497769573913
# services 0.00026918935548377173
# systems 0.00026918935548377173
# office 0.00023073373327180433
# products 0.00023073373327180433
# established 0.00019227811105983694
# including 0.00019227811105983694
# norwegian 0.00019227811105983694
# virgin 0.00019227811105983694
# toronto 0.00019227811105983694
# health 0.00019227811105983694
# management 0.00019227811105983694
# buses 0.00019227811105983694
# capel 0.00019227811105983694

# 把如上格式的txt文件转换为如下格式的txt文件
# company & based & business
# 0.0011152130441470544 & 0.00042301184433164127 & 0.00038455622211967387

def txt_process(input_file_name, output_file_name):
    words = []
    numbers = []
    input_file = input_file_name
    output_file = output_file_name

    with open('output1.txt') as f:
        for line in f:
            word, number = line.split()
            number = str(round(float(number), 6))
            words.append(word + '&')
            numbers.append(number + '&')

    words_1 = words[0:10]
    words_1.insert(0, 'Top 10 &')
    words_1.append('\\\\')
    numbers_1 = numbers[0:10]
    numbers_1.insert(0, '&')
    numbers_1.append('\\\\')

    words_2 = words[10:20]
    words_2.insert(0, 'Bottom 10 &')
    words_2.append('\\\\')
    numbers_2 = numbers[10:20]
    numbers_2.insert(0, '&')
    numbers_2.append('\\\\')
    print(words_1)
    print(words_2)

    print(numbers_1)
    print(numbers_2)
    total_list = [words_1, numbers_1, words_2, numbers_2]
    for line in total_list:
        line_list = ' '.join(map(str, line))
        with open('test_output.txt', 'a') as output_file:
            output_file.write(line_list)
            output_file.write('\n')


txt_process(1, 1)

# import re
#
# output1 = []
# output2 = []
#
# with open('input.txt') as f:
#     for line in f:
#         if re.match(r'Class \d+:', line):
#             current_output = []
#             if output1:
#                 output2 = current_output
#             else:
#                 output1 = current_output
#         else:
#             current_output.append(line)
#
# with open('output1.txt', 'w') as f:
#     f.writelines(output1)
#
# with open('output2.txt', 'w') as f:
#     f.writelines(output2)
