import jittor as jt

a = jt.array([[0,0,1,1], [1,1,2,2], [2,2,3,3]])
b = jt.array([0.1,0.8,0.9])
print(len(a))
d = []
for i in range(len(a)):
    d.append(jt.concat([a[i],b[i]]).unsqueeze(0))
d = jt.concat(d)
print(d)
c = 0.7
jt.misc.nms(d,c)