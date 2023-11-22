import numpy as np

w = np.random.rand(60000*60000)
w = w + np.min(w) + 1
w /= np.sum(w)
np.random.choice(60000*60000, 1000000, p=w)