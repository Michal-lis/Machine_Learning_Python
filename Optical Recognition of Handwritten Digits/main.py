import matplotlib.pyplot as plt

from sklearn import datasets, svm

digits = datasets.load_digits()
# digit DESCR - "Optical Recognition of Handwritten Digits Data Set
# digits images - images of certain digits
# digits target names- [0.1,2,3,4,5,6,7,8,9]
# digits data - numerical representation of targets to learn from

clf = svm.SVC(gamma=0.001, C=100)

print(len(digits.data))

x, y = digits.data[:-1], digits.target[:-1]
clf.fit(x, y)

print("Prediction: ", clf.predict(digits.data[-2].reshape(1, -1)))
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
