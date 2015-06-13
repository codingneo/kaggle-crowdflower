import numpy as np
import pandas as pd

test = pd.read_csv('./data/test.csv').fillna("")
    
# we dont need ID columns
idx = test.id.values.astype(int)


# bagging
# nolearn neural networks
fs1 = './submission/meta-bagging-nn.csv'
fs2 = './submission/meta-bagging-2015-06-07.csv'
fs3 = './submission/svm-2.csv'

preds1 = pd.read_csv(fs1)
preds2 = pd.read_csv(fs2)
preds3 = pd.read_csv(fs3)

# linear average
preds =1.0/3.0*preds1 + 1.0/3.0*preds2 + 1.0/3.0*preds3
preds = preds.astype(int)

# Create your first submission file
submission = pd.DataFrame({"id": idx, "prediction": preds['prediction']})
submission.to_csv("./submission/stacking.csv", index=False)
