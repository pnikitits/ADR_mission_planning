import numpy as np


def make_circle(radius , n_points , center=(0,0,0) , rotation=(0,0,0)):
    """
    Make a circle in 3D space

    Parameters
    ----------
    radius : float
        Radius of the circle
    n_points : int
        Number of points to make the circle
    center : tuple
        Center of the circle
    rotation : tuple
        Rotation of the circle


    Returns
    -------
    np.array
        Array of points on the circle
    """
    theta = np.linspace(0, 2*np.pi, n_points)
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    z = np.zeros_like(theta) + center[2]
    
    # Apply rotation
    rotation_matrix = get_rotation_matrix(rotation)
    points = np.column_stack((x, y, z))
    rotated_points = np.dot(points, rotation_matrix.T)
    
    # convert to list of tuples
    rotated_points = [tuple(point) for point in rotated_points]
    return rotated_points


def get_rotation_matrix(rotation):
    alpha, beta, gamma = rotation
    rotation_matrix = np.array([
        [np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)],
        [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)],
        [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]
    ])
    return rotation_matrix