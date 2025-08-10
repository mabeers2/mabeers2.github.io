import json 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from rotation_matrices import *


def big_box_1(rx,ry,rz):
    """
    Makes truncated pyramid given bounds on size. 
    """
    edges = [(0,1), (2,3),(4,5), (6,7), (0,2), (1,3), (4,6), (5,7), (0,4), (1,5), (2,6), (3,7)]
    if np.random.rand() > .5:
        x_upper = rx/2
        x_lower = (rx/2) * (.2 + .5 * np.random.rand())
    else:
        x_lower = rx/2
        x_upper = (rx/2) * (.2 + .5 * np.random.rand())
    y_lower = 0
    y_upper = ry
    z0 = 0
    z1 = rz * (.4 + .3 * np.random.rand())
    z2 = z1 * (.2 + .2 * np.random.rand())
    z3 = z1 - z2
    pts = [[x_lower, y_lower, z0], 
            [-x_lower, y_lower, z0],
            [x_lower, y_lower, z1],
            [-x_lower, y_lower, z1],
            [x_upper, y_upper, z2],
            [-x_upper, y_upper, z2],
            [x_upper, y_upper, z3],
            [-x_upper, y_upper, z3]]
    pts = np.array(pts)
    return pts, edges

def small_box1(big_box, edges, rx, ry, rz):
    p = .2 + .6 * np.random.rand()
    height = ry * p
    width_at_top = (big_box[4,0] - big_box[5,0]) + (1-p) * (big_box[4,0] - big_box[5,0] - (big_box[0,0] - big_box[1,0]))
    print(width_at_top , "width")
    top_x = (np.abs(width_at_top) / 2) * (.2 + .5 * np.random.rand())
    plane = get_plane(big_box[4], big_box[1], big_box[5])
    top_z = (plane[0] * top_x + plane[1] * height + plane[3]) / (-1 * plane[2])
    low_z = (plane[0] * top_x + plane[1] * 0 + plane[3]) / (-1 * plane[2])
    low_x = big_box[0,0] * (.4 + .55 * np.random.rand())
    # if np.random.rand() < .5:
    if True:
        # lower part of small box meets depth requirements. 
        low_z2 = -1 * (rz - big_box[:, 2].max() + big_box[:, 2].min())
        low_y2 = 0
        low_x2 = (rx / 2)* (.3 + .6 * np.random.rand())
    small_box = np.array([[top_x, height, top_z], 
                            [-top_x, height, top_z],
                            [low_x, 0, low_z],
                            [-low_x, 0, low_z], 
                            [low_x2, low_y2, low_z2], 
                            [-low_x2, low_y2, low_z2]])
    plane2 = get_plane(small_box[0], small_box[2], small_box[4])
    final_x = np.inf
    counter = 0
    max_count = 500
    while (final_x < 0) or (final_x > rx/2):
        final_z = (low_z2 - low_z) * (.3 + .4 * np.random.rand())
        final_y = big_box[:,1].max() * (.3 + .4 * np.random.rand())
        final_x = (plane2[1] * final_y + plane2[2] * final_z + plane2[3]) / (-1 * plane2[0])
        counter += 1
        print(counter)
        if counter == max_count:
            return False, False
    last_points = np.array([[final_x, final_y, final_z], [-final_x, final_y, final_z]])
    small_box = np.vstack((small_box, last_points))
    edges2 = [(0,1), (2,3), (0,2), (1,3), (2,4), (3,5), (4,5),(6,7), (6,4), (7,5), (6,0), (7,1)]
    edges2 = np.array(edges2) + 8
    edges2 = [tuple([i,j]) for i,j in edges2]
    edges_all = edges + edges2
    print("Counter = ", counter)
    if counter < max_count:
        return np.vstack((big_box, small_box)), edges_all
    else:
        return False, False


def make_two_box(rx,ry,rz):
    pts, edges = big_box_1(rx,ry,rz)
    two_box, edges2 = small_box1(pts, edges, rx, ry, rz)
    print("Trying again")
    if edges2 == False:
        return make_two_box(rx, ry,rz)
    else:
        return two_box, edges2

def warp2box(xyz):
    # also shift y-value of either 4,6 or 7,5
    if np.random.rand() > 0.5:
        # shift 4,5
        DY = xyz[:, 1].max() - xyz[:, 1].min()
        side = get_plane(xyz[0], xyz[2], xyz[6])
        # sample point in plane on the same side as line between 0,6 as 4
        z45 = np.random.uniform((xyz[4,2] + xyz[6,2]) / 2, (xyz[12,2] + xyz[0,2]) / 2)
        y45 = np.random.uniform(xyz[4,1], xyz[2,1] + (xyz[4,1] - xyz[2,1]) * 3/4)
        x45 = (side[3] + side[1] * y45 + side[2] * z45) / (-1 * side[0])
        xyz[4] = np.array([x45,y45,z45])
        xyz[5] = xyz[4] * np.array([-1,1,1])
        # adjust 8,9 so that they are in the plane defined by 4,5,0,1
        y89 = np.random.uniform((xyz[4,1] + xyz[2,1]) / 2, (xyz[4,1] + xyz[2,1]) * 3/4)
        side2 = get_plane(xyz[10], xyz[12], xyz[14])
        front = get_plane(xyz[4], xyz[5], xyz[0])
        line_direction = np.cross(side2[: 3], front[:3])
        A = np.array([[side2[1], side2[2]], [front[1], front[2]]])
        B = -1 * np.array([side2[3], front[3]])
        vy,vz = np.linalg.solve(A,B)
        v = np.array([0,vy, vz])
        t = (y89 - vy) / line_direction[1]
        xyz[8] = v + line_direction * t
        xyz[9] = xyz[8] * np.array([-1,1,1])
    else:
        # shift 6,7
        DY = xyz[:, 1].max() - xyz[:, 1].min()
        dy = -1 * (DY * np.random.uniform(low = .2, high = .5))
        xyz[6, 1] += dy
        xyz[7,1] += dy 
        plane = get_plane(xyz[0], xyz[2], xyz[4])
        d_to_plane = plane @ np.hstack((xyz[6], 1))
        xyz[6] -= plane[:3] * d_to_plane
        xyz[7] = xyz[6] * np.array([-1,1,1])
    return xyz


def warp_iterative(xyz, n=1000):
    counter = 0
    while counter < n:
        xyz2 = warp2box(xyz)
        t = (xyz2[9,1] - xyz2[1,1]) / (xyz2[5,1] - xyz2[1,1])
        x9maximum = xyz2[1,0] + (xyz2[5,0] - xyz2[1,0])*t
        t = (xyz2[6,1] - xyz2[2,1]) / (xyz2[4,1] - xyz2[2,1])
        z6minimum = xyz2[2,2] + (xyz2[4,2] - xyz2[2,2])*t
        t = (xyz2[14,1] - xyz2[12,1]) / (xyz2[8,1] - xyz2[12,1])
        z14max = xyz2[12,2] + (xyz2[8,2] - xyz2[12,2]) * t
        if (xyz2[9,0] > x9maximum) & (xyz2[6,2] > z6minimum) & (xyz2[14,2] < z14max) &(xyz[8,0] > xyz[9,0]):
            return xyz2
        counter += 1
    return False

def generate_single_object():
    xyz_warp = False
    while xyz_warp is False:
        xyz,edges = make_two_box(np.random.uniform(1,10), np.random.uniform(1,10), np.random.uniform(1,10))
        xyz_warp = warp_iterative(xyz)
    return xyz_warp   

def generate_single_object2(wx, wy, wz):
    xyz_warp = False
    while xyz_warp is False:
        xyz,edges = make_two_box(wx, wy, wz)
        xyz_warp = warp_iterative(xyz)
    return xyz_warp        


def generate_n_objects(n=100):
    xyzs = []
    while len(xyzs) < n:
        xyz,edges = make_two_box(np.random.uniform(1,10), np.random.uniform(1,10), np.random.uniform(1,10))
        xyz_warp = warp_iterative(xyz)
        if xyz_warp is not False:
            xyzs.append(xyz_warp)
    return xyzs

def generate_n_rotations(n15=10, n30=0, n45=0, n60=0, n75=0):
    Rs = []
    slants = [15,30, 45,60,75]
    quantities = [n15, n30, n45, n60, n75]
    for slant, num in zip(slants, quantities):
        for j in range(num):
            tx = np.random.uniform(20,70) + np.random.choice([0,90,180,270])
            R = rotz(np.random.uniform(0,360)) @ roty(90 - slant) @ rotx(tx)
            Rs.append(R)
    return Rs


def recover_slant(R):
    s = np.degrees(np.arccos(R[2,0]))
    return np.min([s, 180 - s])


def get_plane_ray_intersection(p1,p2,p3, ray):
    plane = get_plane(p1,p2,p3)
    return -1 * (ray * plane[3]) / (plane[:3] @ ray)

def check_interior_1(p1, p2, p3, q):
    u1 = p1 - q
    u1 /= np.linalg.norm(u1)
    u2 = p2 - q
    u2 /= np.linalg.norm(u2)
    u3 = p3 - q
    u3 /= np.linalg.norm(u3)
    angle1 = np.degrees(np.arccos(u1 @ u2))
    angle2 = np.degrees(np.arccos(u1 @ u3))
    angle3 = np.degrees(np.arccos(u2 @ u3))
    return angle1 + angle2 + angle3


def get_triples():
    faces = [[0,1,3,2], [1,3,7,5], [0,2,6,4], [4,5,7,6], [0,1,5,4], [2,3,7,6],
            [8,9,15,14], [12,14,15,13], [10,11,13,12], [10,11,9,8], [11,9,15,13], [8,10,12,14]]
    triples = []
    for face in faces:
        f6 = face.copy() + face[:2]
        for i in range(4):
            triples.append(f6[i:i+3])
    return triples

def check_visible(xyz, index, triples):
    z_stack = []
    ijk_stack = []
    for i, j, k in triples:
        if (i == index) or (j == index) or (k == index):
            continue
        p1 = xyz[i]
        p2 = xyz[j]
        p3 = xyz[k]
        ray = xyz[index]
        q = get_plane_ray_intersection(p1,p2,p3, ray)
        angle_sum = check_interior_1(p1, p2, p3, q)
        if np.isclose(angle_sum, 360):
            # print(f"point {index} occluded by triangle {i}, {j}, {k}")
            z_stack.append(q[2])
            ijk_stack.append((i,j,k))
    if len(z_stack) == 0:
        return True
    # if 0.999 * xyz[index, 2] < min(z_stack): 
    if 0.999 * xyz[index, 2] > max(z_stack):
        return True
    else:
        print(f"point {index} occluded by one of the following triangles")
        print(ijk_stack)
        print(z_stack)
        print("target z= ", xyz[index,2], "\n")
        return False


def scale(xyz):
    xyz2 = xyz.copy()
    xyz2 -= xyz2.mean(axis=0)
    M = np.linalg.norm(xyz2, axis=1).max()
    xyz2 /= M
    center = (xyz2.max(axis=0) + xyz2.min(axis=0)) / 2
    xyz2 -= center
    return xyz2


def num_visible_sym_cors(visible, sym_cors):
    num_sym_cors = 0
    for i,j in sym_cors:
        if i in visible and j in visible:
            num_sym_cors += 1
    return num_sym_cors

def est_slant_given_xyz(xyz):
    u = xyz[0] - xyz[1]
    u /= np.linalg.norm(u)
    theta = np.degrees(np.arccos(u[2]))
    return np.min([theta, 180 - theta])


def plot_dists_4_12(xyz, edges, save=False, size=(10,5)):
    fig, axs = plt.subplots(1,2, figsize=size, sharex=False)
    # dists = [4,12]
    dists = [-4,-12]
    for k in range(2):
        xyzd = scale(xyz) + np.array([0,0,dists[k]])
        visible = [k for k in range(16) if check_visible(xyzd, k, triples)]
        occluded = [k for k in range(16) if k not in visible]
        img = (xyzd[:, :2].T / xyzd[:, 2]).T
        img *= -1
        axs[k].axis("equal")
        axs[k].scatter(img[visible, 0], img[visible, 1], label ='visible')
        axs[k].scatter(img[occluded, 0], img[occluded, 1], alpha = 0.5, label ='occluded')
        for i,j in edges:
            X = [img[i,0], img[j,0]]
            Y = [img[i,1], img[j,1]]
            axs[k].plot(X,Y, color = 'black', alpha=0.5)
        for i in range(img.shape[0]):
            axs[k].text(img[i,0],img[i,1], i, color='blue')
        # axs[k].set_xlim(-1,1)
        # axs[k].set_ylim(-1,1)
        axs[k].legend()
        axs[k].set_title(f"Dist {dists[k]}")
    veridical_slant = est_slant_given_xyz(xyz)
    fig.suptitle(f"Veridical Slant = {round(veridical_slant)}")
    fig.tight_layout()
    if save:
        plt.savefig(save, dpi = 300)
    else:
        fig.show()


def plot_ars(xz, xy, yz, size=(10,4), save=False):
    ars = [xz, xy, yz]
    titles = [r"$log_{10}(x/z)$", r"$log_{10}(x/y)$", r"$log_{10}(y/z)$"]
    fig, axs = plt.subplots(1,3, figsize=size, sharex= True, sharey=True)
    for i in range(3):
        axs[i].hist(np.log10(ars[i]))
        axs[i].set_title(titles[i])
        axs[i].set_xlabel(titles[i])
    axs[0].set_ylabel("Frequency")
    fig.tight_layout()
    if save:
        plt.savefig(save, dpi = 300)
    else:
        fig.show()


def get_R(xyz):
    vector2D = (xyz[0] - xyz[1])[:2]
    vector2D /= np.linalg.norm(vector2D)
    Rz1 = rot_axis(np.array([0,0,1]),  np.degrees(np.arccos(vector2D[0])))
    Rz2 = rot_axis(np.array([0,0,1]),  -1 * np.degrees(np.arccos(vector2D[0])))
    xyzR1 = (Rz1 @ xyz.T).T
    if np.isclose(xyzR1[0,1], xyzR1[1,1]):
        return Rz1
    else:
        return Rz2

def get_extreme_n(n_small, n_large, slant, rats, target_slant):
    i = np.where(target_slant == slant)[0]
    j = np.argsort(rats[i])
    return np.hstack((i[j[:n_small]], i[j[-n_large:]]))




triples = get_triples()
sym_cors = [(i,i+1) for i in range(0,16,2)]
xyzs100 = []
target_slants = [15]*5 + [30]*5
# target_slants = [15]*10 + [30]*10
np.random.shuffle(target_slants)
i = 0
nrots = 50
xz = []
yz = []
xy = []
while len(xyzs100) < 10: # 100
    wx = np.random.uniform(1,16)
    wy = np.random.uniform(1,16) # 4
    wz = np.random.uniform(1,16)
    # candidate_xyz = scale(generate_single_object())
    candidate_xyz = scale(generate_single_object2(wx, wy, wz))
    j = 0
    while j < nrots:
        tx = np.random.uniform(20,70) + np.random.choice([0,90,180,270])
        R = rotz(np.random.uniform(0,360)) @ roty(90 - target_slants[i]) @ rotx(tx) 
        xyzR = (R @ candidate_xyz.T).T
        xyzR4 = xyzR + np.array([0,0,-4])
        xyzR12 = xyzR + np.array([0,0,-12])
        # xyzR4 = xyzR + np.array([0,0,4])
        # xyzR12 = xyzR + np.array([0,0,12])
        visible4 = [k for k in range(16) if check_visible(xyzR4, k, triples)]
        visible12 = [k for k in range(16) if check_visible(xyzR12, k, triples)]
        rx,ry,rz = candidate_xyz.max(axis=0) - candidate_xyz.min(axis=0)
        rat_xz = rx/rz
        rat_yz = ry/rz
        rat_xy = rx/ry
        if (num_visible_sym_cors(visible4, sym_cors) >= 5) and (num_visible_sym_cors(visible12, sym_cors) >= 5) & (0 < rat_xz) & (rat_xz < 16)& (0 < rat_yz) & (rat_yz < 16)& (0 < rat_xy) & (rat_xy < 16):
            xyzs100.append(xyzR)  
            xz.append(rat_xz)
            yz.append(rat_yz)
            xy.append(rat_xy)
            i += 1
            break          
        j += 1
        print(f"{j} / {nrots} Done")
    print(f"i= {i}, len(xyzs) = {len(xyzs100)}")



# slant15_idx = get_extreme_n(5,5,15,np.array(xz), np.array(target_slants))
# slant30_idx = get_extreme_n(5,5,30,np.array(xz), np.array(target_slants))
# xyz_top20 = [xyzs100[k] for k in np.hstack((slant15_idx,slant30_idx))]
# vsa_top20 = [target_slants[k] for k in np.hstack((slant15_idx,slant30_idx))]
# order = np.repeat(np.arange(20), 5)
# np.random.shuffle(order)
# dists = np.zeros_like(order)
# for i in range(20):
#     dist5 = np.array([4,6,8,10,12])
#     np.random.shuffle(dist5)
#     dists[order == i] = dist5
    
dataset = {"xyz":[xyz.tolist() for xyz in xyzs100], 
              'vsa':target_slants, 
              "R":[get_R(xyz).tolist() for xyz in xyzs100], 
              "triples":np.ravel(triples).tolist()}
json_object = json.dumps(dataset, indent = 0)
with open(f"./demo_dataset.json", "w") as handle:
    handle.write(json_object)

