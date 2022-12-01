from reprint import output
import time
import numpy as np

def showBox():
    with output(output_type='dict') as out:
        for i in range(100):
            out[1] = "+---+\n"
            out[2] = "|" + str(i) + " |\n"
            out[3] = "+---+\n"
            time.sleep(0.05)

#showBox()

s = [1,2,3,4,5,6,7,8,9]
s = np.array(s)
print(s.shape)
s = s.reshape(3,3)
print(s.shape)

p = np.array([[0.22183608, 0.23086475, 0.27100843, 0.27629077]], dtype=np.float32)
i = np.argmax(p[0])
print(i)
print(p[0][i])
print(p[0].shape)

q = np.array([0.25]*4)
print(q)
print(q.shape)