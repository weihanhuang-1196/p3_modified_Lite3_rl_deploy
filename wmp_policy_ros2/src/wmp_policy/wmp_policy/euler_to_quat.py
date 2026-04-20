import numpy as np

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ]

def axis_angle(axis, deg):
    rad = np.deg2rad(deg)
    s = np.sin(rad/2)
    c = np.cos(rad/2)

    if axis=="x":
        return [c,s,0,0]
    if axis=="y":
        return [c,0,s,0]
    if axis=="z":
        return [c,0,0,s]

q = [1,0,0,0]

q = quat_mul(axis_angle("x",-90), q)  # roll
q = quat_mul(axis_angle("z",90), q)   # yaw
q = quat_mul(axis_angle("y",10), q)   # pitch
import numpy as np

list_name = [["w","x","y","z"]]
arr = np.array(list_name)


print(q)