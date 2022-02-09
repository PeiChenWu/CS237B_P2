#!/usr/bin/env python

import cvxpy as cp
import numpy as np
import pdb  

from utils import *

def solve_socp(x, As, bs, cs, ds, F, g, h, verbose=False):
    """
    Solves an SOCP of the form:

    minimize(h^T x)
    subject to:
        ||A_i x + b_i||_2 <= c_i^T x + d_i    for all i
        F x == g

    Args:
        x       - cvx variable.
        As      - list of A_i numpy matrices.
        bs      - list of b_i numpy vectors.
        cs      - list of c_i numpy vectors.
        ds      - list of d_i numpy vectors.
        F       - numpy matrix.
        g       - numpy vector.
        h       - numpy vector.
        verbose - whether to print verbose cvx output.

    Return:
        x - the optimal value as a numpy array, or None if the problem is
            infeasible or unbounded.
    """
    objective = cp.Minimize(h.T @ x)
    constraints = []
    for A, b, c, d in zip(As, bs, cs, ds):
        constraints.append(cp.SOC(c.T @ x + d, A @ x + b))
    constraints.append(F @ x == g)
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)

    if prob.status in ['infeasible', 'unbounded']:
        return None

    return x.value

def grasp_optimization(grasp_normals, points, friction_coeffs, wrench_ext):
    """
    Solve the grasp force optimization problem as an SOCP. Handles 2D and 3D cases.

    Args:
        grasp_normals   - list of M surface normals at the contact points, pointing inwards.
        points          - list of M grasp points p^(i).
        friction_coeffs - friction coefficients mu_i at each point p^(i).
        wrench_ext      - external wrench applied to the object.

    Return:
        f
    """
    D = points[0].shape[0]  # planar: 2, spatial: 3
    N = wrench_size(D)      # planar: 3, spatial: 6
    M = len(points)
    transformations = [compute_local_transformation(n) for n in grasp_normals]

    ########## Your code starts here ##########
    As = []
    bs = []
    cs = []
    ds = []

    for i in range(M): # friction cone constraints
        A = np.zeros((D,D*M+1))
        A[0,D*i] = 1
        if D==3:
            A[1,D*i+1] = 1
        As.append(A)
        b = np.zeros((D))
        bs.append(b)
        c = np.zeros((D*M+1))
        if D==2:
            c[D*i+1] = friction_coeffs[i]
        elif D==3:
            c[D*i+2] = friction_coeffs[i]
        cs.append(c)
        d = 0
        ds.append(d)

    for i in range(M): # less than s constraints
        A = np.zeros((D,D*M+1))
        A[0,D*i] = 1
        A[1,D*i+1] = 1
        if D==3:
            A[2,D*i+2] = 1
        As.append(A)
        b = np.zeros((D))
        bs.append(b)
        c = np.zeros((D*M+1))
        c[-1] = 1
        cs.append(c)
        d = 0
        ds.append(d)
        
    g = np.zeros((N))
    g = -wrench_ext

    F = np.zeros((N,D*M+1))
    for i in range(M):
        F[0:D,i*D:i*D+D] = transformations[i]
        skew_p = cross_matrix(points[i]) # cross_matrix from util.py
        F[D:,i*D:i*D+D] = skew_p@transformations[i]    
        
    h = np.zeros((D*M+1))
    h[-1] = 1
    x = cp.Variable(D*M+1)
        
    x = solve_socp(x, As, bs, cs, ds, F, g, h, verbose=False)

    # TODO: extract the grasp forces from x as a stacked 1D vector
    f = x[:-1]

    ########## Your code ends here ##########

    # Transform the forces to the global frame
    F = f.reshape(M,D)
    forces = [T.dot(f) for T, f in zip(transformations, F)]


    return forces

def precompute_force_closure(grasp_normals, points, friction_coeffs):
    """
    Precompute the force optimization problem so that force closure grasps can
    be found for any arbitrary external wrench without redoing the optimization.

    Args:
        grasp_normals   - list of M contact normals, pointing inwards from the object surface.
        points          - list of M contact points p^(i).
        friction_coeffs - friction coefficients mu_i at each point p^(i).

    Return:
        force_closure(wrench_ext) - a function that takes as input an external wrench and
                                    returns a set of forces that maintains force closure.
    """
    D = points[0].shape[0]  # planar: 2, spatial: 3
    N = wrench_size(D)      # planar: 3, spatial: 6
    M = len(points)

    ########## Your code starts here ##########
    # TODO: Precompute the optimal forces for the 12 signed unit external
    #       wrenches and store them as rows in the matrix F. This matrix will be
    #       captured by the returned force_closure() function.
    F = np.zeros((2*N, M*D))
    
    w_basis = []
    w_basis.append(np.array([1,0,0,0,0,0]))
    w_basis.append(np.array([-1,0,0,0,0,0]))
    w_basis.append(np.array([0,1,0,0,0,0]))
    w_basis.append(np.array([0,-1,0,0,0,0]))
    w_basis.append(np.array([0,0,1,0,0,0]))
    w_basis.append(np.array([0,0,-1,0,0,0]))
    w_basis.append(np.array([0,0,0,1,0,0]))
    w_basis.append(np.array([0,0,0,-1,0,0]))
    w_basis.append(np.array([0,0,0,0,1,0]))
    w_basis.append(np.array([0,0,0,0,-1,0]))
    w_basis.append(np.array([0,0,0,0,0,1]))
    w_basis.append(np.array([0,0,0,0,0,-1]))
    
    for i in range(0,2*N,2):
        F[i,:] = np.array(grasp_optimization(grasp_normals, points, friction_coeffs, w_basis[i][:N])).reshape(M*D) 
        F[i+1,:] = np.array(grasp_optimization(grasp_normals, points, friction_coeffs, w_basis[i+1][:N])).reshape(M*D) 

    ########## Your code ends here ##########

    def force_closure(wrench_ext):
        """
        Return a set of forces that maintain force closure for the given
        external wrench using the precomputed parameters.

        Args:
            wrench_ext - external wrench applied to the object.

        Return:
            f - grasp forces as a list of M numpy arrays.
        """

        ########## Your code starts here ##########
        # TODO: Compute the force closure forces as a stacked vector of shape (M*D)
        f = np.zeros(M*D)

        for i,w in enumerate(wrench_ext): 
            f += np.max((0,w))*F[2*i,:] + np.max((0,-w))*F[2*i+1,:]
        
        ########## Your code ends here ##########

        forces = [f_i for f_i in f.reshape(M,D)]
        return forces

    return force_closure
