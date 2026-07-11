import numpy as np

d = np.load('/home/amine/PycharmProjects/osunator/features/0a6f180152473b4a863afb0c5f9f4fec.npz')
p = d['approach_progress']
print(p.min(), p.max(), np.isnan(p).any(), (p == 0).mean(), (p == 1).mean())