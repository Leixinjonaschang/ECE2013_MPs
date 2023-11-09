import matplotlib.pyplot as plt

pre_no_priors = [0.0, 1.0, 1.0, 1.0, 0.9130434782608695, 0.8333333333333334, 0.625, 0.918918918918919, 1.0, 1.0,
                 0.9166666666666666, 0.7777777777777778, 0.8222222222222222, 0.5409836065573771]
recall_no_priors = [0.0, 0.8043478260869565, 0.09523809523809523, 1.0, 0.9545454545454546, 0.9375, 1.0, 1.0, 0.75, 0.9,
                    0.9777777777777777, 1.0, 0.9736842105263158, 0.9428571428571428]
f1_no_priors = [0.0, 0.891566265060241, 0.17391304347826084, 1.0, 0.9333333333333332, 0.8823529411764706,
                0.7692307692307693, 0.9577464788732395, 0.8571428571428571, 0.9473684210526316, 0.946236559139785,
                0.8750000000000001, 0.891566265060241, 0.6875]

pre = [0.0, 1.0, 0.0, 1.0, 0.9545454545454546, 0.8461538461538461, 0.5918367346938775, 0.8947368421052632, 1.0, 1.0,
       0.9166666666666666, 0.75, 0.8222222222222222, 0.5076923076923077]
recall = [0.0, 0.7608695652173914, 0.0, 1.0, 0.9545454545454546, 0.9166666666666666, 0.9666666666666667, 1.0, 0.625,
          0.9, 0.9777777777777777, 1.0, 0.9736842105263158, 0.9428571428571428]
f1 = [0.0, 0.8641975308641976, 0.0, 1.0, 0.9545454545454546, 0.8799999999999999, 0.7341772151898733, 0.9444444444444444,
      0.7692307692307693, 0.9473684210526316, 0.946236559139785, 0.8571428571428571, 0.891566265060241,
      0.6599999999999999]

diff_pre = []
diff_recall = []
diff_f1 = []

for i in range(14):
    diff_pre.append(pre[i] - pre_no_priors[i])
    diff_recall.append(recall[i] - recall_no_priors[i])
    diff_f1.append(f1[i] - f1_no_priors[i])
print('diff_pre', diff_pre)
print('diff_recall', diff_recall)
print('diff_f1', diff_f1)

# Plot

x = []
for i in range(14):
    x.append(i)

# plt.plot(x, pre_no_priors, label='pre_no_priors')
# plt.plot(x, pre, label='pre')
#
# plt.plot(x, recall_no_priors, label='recall_no_priors')
# plt.plot(x, recall, label='recall')
#
# plt.plot(x, f1_no_priors, label='f1_no_priors')
# plt.plot(x, f1, label='f1')
width = 0.15

for i in range(len(x)):
    x[i] = x[i] - 2 * width

plt.bar(x, pre_no_priors, width=width, label='pre_no_priors')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, pre, width=width, label='pre')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, recall_no_priors, width=width, label='recall_no_priors')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, recall, width=width, label='recall')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, f1_no_priors, width=width, label='f1_no_priors')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, f1, width=width, label='f1')

plt.legend(['pre_no_priors', 'pre', 'recall_no_priors', 'recall', 'f1_no_priors', 'f1'])
plt.legend(loc='lower left')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
           ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'])

plt.show()
