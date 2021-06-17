# Glass-Classification
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python# For example,here 's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing,cSv file I/o (e.g. pd. read_csv)
# Input data files are available in the read-only "../input/" directory
# For example，running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname,-, filenames in os.walk ( ' /kaggle/input' ):
  for filename in filenames :
    print( os.path.join(dirname，filename ) )
    
# You can write up to 56B to the current directory (/kaggle /working/) that gets preserved as output when you create a version using"Save &Run All"
# You can also write temporary files to /kaggle/temp/， but they won't be saved outside of the current session

First we will import Pandas lib
import pandas as pd
reading the head of the data set.
df=pd .read_csv( ' ../input/glass/glass.csv ' )
df.head()
now.we can import some visulization library like matplotlib and seaborn and also we willimport numpy.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

first we have to understand the data.

df.info()

NO, null values are there,that is a good sign.

df.shape
sns.countplot(df['Type'])

scaler=standardScaler()

Type Markdown and LaTex: a2

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x, y ,test_size=.35, random_state=101)

x_train.shape

x_test.shape

x_train=scaler.fit_transform( x_train)x_test=scaler.transform(x_test)

from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeclassifier()

model.fit(X_train,y_train)

prd=model.predict(×_test)

prd_train=model.predict(x_train)

from sklearn.metrics import classification_report
print(classification_report(prd , y_test) )
print(classification_report(prd_train, y_train ))



we are not achieving much high accuracy with these models ,we can try for DNN next.


