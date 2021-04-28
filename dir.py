import os

a = 'c:/nmb/nmb_data/etc/M2.wav'

b = os.path.split(a)
c = os.path.join('c:/nmb/nmb_data/etc/', 'M2.wav')
d = os.path.splitext(c)
print(b[0])
print(c)
print(d[0][-1])