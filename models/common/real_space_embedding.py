import numpy as np

from .preprocessing import *


def make_real_space_embedding_model(detector_data, bpsplit_z, bpsplit_phi, normalize=False):
    '''
    Returns an object which behaves the same way as a keras embeding model, but internally consists of a dictionary lookup. It also has the same output shape.
    
    Parameters:
    * detector_data: pd dataframe
    * bpsplit_z (ndarray): bounds of z-splitting
    * bpsplit_phi (int): phi splitting
    * [OPT] normalize (bool): normalize data
    
    Returns:
    * Callable
    '''
    assert bpsplit_phi >= 1
    assert len(bpsplit_z) >= 2
    
    total_beampipe_split = (len(bpsplit_z)-1)*bpsplit_phi
    total_node_num = len(detector_data.index) - 1 + total_beampipe_split
    
    # here create unique positions for all surfaces originating from the beampipe split. 
    # The z-coords are the same for bpsplit_phi elements, so we need to add small offsets in x-y direction.
    angles = np.linspace(0,2*np.pi,bpsplit_phi,endpoint=False)
    
    phi_factor = 0 if bpsplit_phi == 1 else 0.0001
    
    bp_positons = np.vstack([
        np.resize([ phi_factor*np.cos(a) for a in angles ], total_beampipe_split),
        np.resize([ phi_factor*np.sin(a) for a in angles ], total_beampipe_split),
        np.sort(np.resize(positions_from_bounds(bpsplit_z),total_beampipe_split))
    ]).transpose()
    
    detector_positions = detector_data[['x','y','z']].to_numpy()[1:]
    
    positions = np.vstack([bp_positons, detector_positions]) 
    ids = np.arange(total_node_num)
    
    if normalize:
        positions /= np.amax(positions)
    
    assert len(positions) == len(np.unique(positions, axis=0))
    assert len(ids) == len(positions)
    
    id_position_map = { i: pos for i, pos in zip(ids, positions) }
    return lambda ids: np.expand_dims(np.vstack([ id_position_map[i] for i in ids ]),1)
