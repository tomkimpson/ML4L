import pandas as pd
from pandas_tfrecords import pd2tf, tf2pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c'], 'C': [[1, 2], [3, 4], [5, 6]]})

print (df)

# local
print ('df to trf:')
pd2tf(df, './tfrecords',compression_type=None)

print('load trf:')
my_df = tf2pd('./tfrecords')
print(my_df)
