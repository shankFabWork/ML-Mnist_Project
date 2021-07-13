import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.datasets import fetch_openml
dataset = fetch_openml('mnist_784')

X = dataset.data.values
Y = dataset.target

Y = Y.astype('int32')

# some_digit = X[0]
# some_digit_image = some_digit.reshape(28,28)

# plt.imshow(some_digit_image,"binary")
# plt.axis("off")
# plt.show()


# %matplotlib qt
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     im = X[i]
#     im = im.reshape(28,28)
#     plt.imshow(im,"binary")
#     plt.xlabel("Label : {}".format(Y[i]))
# plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)

from sklearn.linear_model import LogisticRegression   
lr = LogisticRegression()

lr.fit(X_train,Y_train)

print(lr.score(X_train,Y_train) )
print(lr.score(X_test,Y_test) )

Y_pred = lr.predict(X_test)

Y_test = Y_test.astype('int32').values



for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    im = X_test[i]
    im = im.reshape(28,28)
    plt.imshow(im,"binary")     
    plt.xlabel("Actual Label : {} \n Predicted Label : {}".format(Y_test[i],Y_pred[i]))
plt.show()

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()

dtc.fit(X_train,Y_train)
print(dtc.score(X_train,Y_train) )
print(dtc.score(X_test,Y_test) )


