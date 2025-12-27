import jittor as jt

a = jt.array([1,2,3,4,5,6], dtype=jt.float32)
print(len(a.shape))
if len(a.shape)==1:
    a = a.unsqueeze(0)
b = a[:,5:6]
print(b)