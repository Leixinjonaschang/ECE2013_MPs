import re

# output = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
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
# for i, output in zip(range(1, 15), output):
#     with open(f'output_class_{i}_features', 'w') as f:
#         f.writelines(output)
# with open('output1.txt', 'w') as f:
#     f.writelines(output1)
#
# with open('output2.txt', 'w') as f:
#     f.writelines(output2)
import re


def txt_process(input_file_name, output_file_name):
    words = []
    numbers = []

    with open(input_file_name) as f:
        for line in f:
            if line.strip():
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
    # print(words_1)
    # print(words_2)
    #
    # print(numbers_1)
    # print(numbers_2)
    total_list = [words_1, numbers_1, words_2, numbers_2]
    for line in total_list:
        line_list = ' '.join(map(str, line))
        with open(output_file_name, 'a') as output_file:
            output_file.write(line_list)
            output_file.write('\n')


# output_files = []
# current_output = None
#
# with open('top_features.txt') as f:
#     for line in f:
#         if re.match(r'Class \d+:', line):
#             current_output = []
#             output_files.append(current_output)
#         else:
#             current_output.append(line)
#
# for i, output in enumerate(output_files, 1):
#     with open('input_' + str(i) + '.txt', 'w') as f:
#         f.writelines(output)

for i in range(1, 15):
    txt_process(f'input_{i}.txt', f'output_class_{i}_features.txt')
# txt_process('input_1.txt', 'output_class_1_features.txt')
