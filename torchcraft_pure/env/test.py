import numpy as np
from sklearn.preprocessing import OneHotEncoder

obs_n = [
    [0, 0, 1],
    [0, 2, 1],
    [1, 1, 2],
    [1, 2, 3],
]
obs_n = np.array(obs_n, dtype=np.float32)
onehot_encoder = OneHotEncoder(sparse=False)
category_pro = onehot_encoder.fit_transform(obs_n[:, [0, 1]])
print(category_pro)
print(np.concatenate(np.concatenate([category_pro, obs_n[:, 2:]], axis=1)))