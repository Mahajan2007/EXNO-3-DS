## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
## 1. Ordinal Encoding
```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("data.csv")

ord1_order = ['Cold', 'Warm', 'Hot', 'Very Hot']
ord2_order = ['High School', 'Diploma', 'Bachelors', 'Masters', 'PhD']

ordinal_enc = OrdinalEncoder(categories=[ord1_order, ord2_order])
df[['Ord_1_encoded', 'Ord_2_encoded']] = ordinal_enc.fit_transform(df[['Ord_1', 'Ord_2']])

print(df[['Ord_1', 'Ord_1_encoded', 'Ord_2', 'Ord_2_encoded']])
```
<img width="561" height="250" alt="image" src="https://github.com/user-attachments/assets/13b6c5a7-d6dd-4af1-9881-28c9e6f1d669" />

## 2. Label Encoding

```Python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data.csv")

le_city = LabelEncoder()
df['City_label'] = le_city.fit_transform(df['City'])

print(df[['City', 'City_label']])
```
<img width="280" height="246" alt="image" src="https://github.com/user-attachments/assets/ad6dffca-4d59-4668-93f6-38efe803af1f" />

## 3. Binary Encoding (Manual)
```Python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data.csv")

# Label encode first
le_city = LabelEncoder()
df['City_label'] = le_city.fit_transform(df['City'])

n_unique = df['City_label'].nunique()
max_bits = int(np.ceil(np.log2(n_unique)))
for i in range(max_bits):
    df[f'City_bin_{i}'] = df['City_label'].apply(lambda x: (x >> i) & 1)

print(df[['City', 'City_label'] + [f'City_bin_{i}' for i in range(max_bits)]])
```
<img width="526" height="251" alt="image" src="https://github.com/user-attachments/assets/f334a90a-ab70-4c72-a167-3911eb9e5cd1" />

## 4. One Hot Encoding
```Python
import pandas as pd

df = pd.read_csv("data.csv")

df_ohe = pd.get_dummies(df['City'], prefix='City')
df = pd.concat([df, df_ohe], axis=1)

print(df[['City'] + list(df_ohe.columns)])

```
<img width="702" height="245" alt="image" src="https://github.com/user-attachments/assets/55620a76-f829-459a-901b-b0bc99697a4f" />

## 5. Log Transformation
```Python
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

df['Target_log'] = np.log(df['Target'] + 1)
print(df[['Target', 'Target_log']])
```
<img width="236" height="242" alt="image" src="https://github.com/user-attachments/assets/cb9edede-84ae-4674-9922-37a045aeac45" />


## 6. Reciprocal Transformation
```Python
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

df['Target_reciprocal'] = 1 / (df['Target'] + 1e-6)
print(df[['Target', 'Target_reciprocal']])
```
<img width="340" height="247" alt="image" src="https://github.com/user-attachments/assets/9ef64f79-70aa-4d57-a1df-52d085bae1b1" />


## 7. Square Root Transformation
```Python
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

df['Target_sqrt'] = np.sqrt(df['Target'])
print(df[['Target', 'Target_sqrt']])
```
<img width="293" height="252" alt="image" src="https://github.com/user-attachments/assets/fc31c054-6a4c-4193-a8c8-e3a092094050" />

## 8. Square Transformation
```Python
import pandas as pd

df = pd.read_csv("data.csv")

df['Target_square'] = df['Target'] ** 2
print(df[['Target', 'Target_square']])
```
<img width="307" height="247" alt="image" src="https://github.com/user-attachments/assets/662a82dd-eebd-4bc2-a6ed-716ae9cd21e4" />

## 9. Boxcox Method
```Python
import pandas as pd
from sklearn.preprocessing import PowerTransformer

df = pd.read_csv("data.csv")

# Boxcox requires strictly positive values, so add 1
pt_boxcox = PowerTransformer(method='box-cox')
df['Target_boxcox'] = pt_boxcox.fit_transform(df[['Target']] + 1)

print(df[['Target', 'Target_boxcox']])
```
<img width="301" height="260" alt="image" src="https://github.com/user-attachments/assets/6dedeb96-c54c-4e14-a18c-1f1e0a2fc1a7" />

## 10. Yeo-Johnson Method
```Python
import pandas as pd
from sklearn.preprocessing import PowerTransformer

df = pd.read_csv("data.csv")

pt_yeojohnson = PowerTransformer(method='yeo-johnson')
df['Target_yeojohnson'] = pt_yeojohnson.fit_transform(df[['Target']])

print(df[['Target', 'Target_yeojohnson']])
```
 <img width="365" height="243" alt="image" src="https://github.com/user-attachments/assets/6e18a136-2dfb-4548-9a12-9a89eaa5ae97" />
  
# RESULT:
The dataset’s categorical features can be converted to numerical form using various encoding techniques such as ordinal encoding, which preserves the order among categories, label encoding, which assigns each category a unique integer, binary encoding, which represents categories with binary digits, and one-hot encoding, which creates separate columns for each category to indicate presence or absence. For numerical features, different data transformation techniques like log, reciprocal, square root, and square transformations can be used to reshape distributions, while advanced power transformations such as Box-Cox and Yeo-Johnson help in normalizing skewed data. These preprocessing steps are essential for improving the performance and interpretability of machine learning models.
