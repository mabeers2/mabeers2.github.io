import numpy as np 
import matplotlib.pyplot as plt

def rotx(theta):
    """
    Input: theta (degrees)
    Output: 3x3 Rotation matrix encoding rotation of theta degrees about X axis
    """
    theta = np.radians(theta)
    s = np.sin(theta)
    c = np.cos(theta)
    M = np.array([[1,0,0], [0,c,-s],[0,s,c]])
    return M

def roty(theta):
    """
    Input: theta (degrees)
    Output: 3x3 Rotation matrix encoding rotation of theta degrees about Y axis
    """
    theta = np.radians(theta)
    s = np.sin(theta)
    c = np.cos(theta)
    M = np.array([[c,0,s], [0,1,0],[-s,0,c]])
    return M

def rotz(theta):
    """
    Input: theta (degrees)
    Output: 3x3 Rotation matrix encoding rotation of theta degrees about Z axis
    """
    theta = np.radians(theta)
    s = np.sin(theta)
    c = np.cos(theta)
    M = np.array([[c,-s,0], [s,c,0],[0,0,1]])
    return M

def rot_axis(u, theta):
    """
    u: axis of rotation
    theta: degrees of rotation
    See section titled *Rotation matrix from axis and angle*
    https://en.wikipedia.org/wiki/Rotation_matrix
    Useful when dealing with intrinsic euler angles, which are the default 
    angles used in Three.js
    """
    theta = np.radians(theta)
    cross = np.array([[0,-u[2], u[1]], 
                        [u[2], 0, -u[0]], 
                        [-u[1], u[0], 0]])
    R = np.cos(theta) * np.eye(3) + np.sin(theta)*cross + (1- np.cos(theta)) * np.outer(u,u)
    return R


def expand(xyz, triples, center = False, scale = False):
    new = []
    for i,j,k in triples:
        new.extend([xyz[i], xyz[j], xyz[k]])
    new = np.vstack(new)
    if center:
        new -= new.mean(axis=0)
    if scale:
        M = np.linalg.norm(new, axis=1).max()
        new = new/M
    return new


def get_plane(p1, p2, p3):
    u = p1 - p2
    u = u / np.linalg.norm(u)
    v = p3 - p2
    v = v/np.linalg.norm(v)
    w = np.cross(u,v)
    w = w / np.linalg.norm(w)
    d = -1 * p2 @ w
    return np.hstack((w, d))



def plot3(xyz, points=False, edges=False, aspect=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if points:
        ax.scatter(xyz[:, 0], xyz[:,1], xyz[:,2])
    else:
        for i in range(xyz.shape[0]):
            ax.text(*xyz[i], i, color='blue')
    if edges:
        for edge in edges:
            x = xyz[[edge[0], edge[1]], 0]
            y = xyz[[edge[0], edge[1]], 1]
            z = xyz[[edge[0], edge[1]], 2]
            ax.plot3D(x,y,z, color='blue')
    if aspect == None: 
        rng = xyz.max(axis=0) - xyz.min(axis=0)
        ax.set_box_aspect(list(rng))
    else:
        ax.set_box_aspect(aspect)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.show()

