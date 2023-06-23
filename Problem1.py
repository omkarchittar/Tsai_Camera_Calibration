import scipy as sp
import numpy as np

world_pts = [[0,0,0,1], [0,3,0,1], [0,7,0,1], [0,11,0,1], [7,1,0,1], [0,11,7,1], [7,9,0,1], [0,1,7,1]]
image_pts = [(757,213), (758,415), (758,686), (759,966), (1190,172), (329,1041), (1204,850), (340,159)]
homo_mat = []
proj_pts = []
reproj_error = []
error_mean = error = error_tot = 0
n = len(world_pts)

# converting to homogenous system of linear equations
for point in range(n):
    u = image_pts[point][0]
    v = image_pts[point][1]
    X = world_pts[point][0]
    Y = world_pts[point][1]
    Z = world_pts[point][2]
    first = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
    second = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]
    homo_mat.append(first)
    homo_mat.append(second)

[U,S,Vt] = np.linalg.svd(homo_mat)
P = Vt[-1,:]
P = np.reshape(P,(3,4))

print("\nCamera Projection Matrix: \n", P)

# Using Gram schmidt method to transform a set of linearly independent vectors into a set of orthonormal vectors that span the same space spanned by the original set.
M = P[0:3, 0:3]                 # using the 3x3 square matrix from the projection matrix
R, Q = sp.linalg.rq(M)          # RQ decomposition of the extracted matrix

# Intrinsic Matrix
K = R
print("\nCamera Intrinsic Matrix: \n", K)

# Extrinsic Matrix
E = np.matmul(np.linalg.inv(R), P)
print("\nExtrinsic Matrix: \n", E)

# Rotation component
R = E[:, 0:3]
print("\nRotation: \n", R)

# Translation component
T = E[:, -1]
print("\nTranslation: \n", T, "\n")

# Calculating the Reprojection Error
for point in range(len(world_pts)):
    proj_image = np.matmul(P, world_pts[point])
    u = proj_image[0]/proj_image[2]
    v = proj_image[1]/proj_image[2]
    new_image = [u, v]
    proj_pts.append(new_image)
    # calculating error using Euclidean distance
    error = np.sqrt((new_image[0]-image_pts[point][0])**2 + (new_image[1]-image_pts[point][1])**2)
    reproj_error.append(error)
    
for i in range(len(reproj_error)):
    print("Reproj error for point ", i+1 , " is: ", reproj_error[i])
    error_tot += reproj_error[i]
print()

error_mean = error_tot/n
print("Mean Error: \n", error_mean)


