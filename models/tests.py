import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from common.preprocessing import *
from common.real_space_embedding import *
from plot_embedding import *




def old_uniform_beampipe_split(prop_data, z_split, phi_split, return_z_distribution=False):
    '''
    NOTE: This was originally part of the main preprocessing pipeline, but is now used as a test for the preprocessing pipeline
    
    Maps all GeoIDs which are 0 to new numbers in [0, z_split * phi_split], dependent on their track parameters. 
    
    Parameters:
    * prop_data: pandas dataframe containing the propagation data
    * z_split: in how many parts the beampipe will be split in z-direction
    * phi_split: in how many parts the beampipe will be split angle wise
    * [OPT] return_z_distribution: if true, return z_distribution
    
    Returns:
    * pandas dataframe containing modified propagation data
    '''
    
    assert z_split > 0 and phi_split > 0
    assert np.amin( prop_data[ prop_data['start_id'] != 0 ]['start_id'].to_numpy() ) >= z_split*phi_split
    
    # Get z positions 
    bp_z_positions = prop_data[ prop_data['start_id'] == 0 ]['pos_z'].to_numpy()
    
    # Compute bin size. increase the z-range slightly to avoid border effects
    z_bin_size = ( np.amax(bp_z_positions) - np.amin(bp_z_positions) )*1.01 / z_split
    
    assert bp_z_positions.all() >= 0 and bp_z_positions.all() < np.amax(bp_z_positions)
    
    # Translate z coords to be all > 0, then divide by bin_size and cast to int
    bp_new_ids = ((bp_z_positions - np.amin(bp_z_positions)) / z_bin_size).astype(np.uint64)
    
    # For conditional return
    z_ids, z_ids_weights = np.unique(bp_new_ids, return_counts=True)
    
    # expand new ids so that there is space for phi-splitting
    bp_new_ids *= phi_split
    
    # then do phi-splitting
    bp_dir_x = prop_data[ prop_data['start_id'] == 0 ]['dir_x'].to_numpy()
    bp_dir_y = prop_data[ prop_data['start_id'] == 0 ]['dir_y'].to_numpy()
    
    # get angle and normalize to [0, 2pi]
    bp_phi_angles = np.arctan2(bp_dir_x, bp_dir_y) + np.pi
    assert bp_phi_angles.all() >= 0 and bp_phi_angles.all() < 2*np.pi
    
    phi_bin_size = 2*np.pi / phi_split
    
    bp_new_ids += (bp_phi_angles / phi_bin_size).astype(np.uint64)
    
    prop_data.loc[ prop_data['start_id'] == 0, 'start_id'] = bp_new_ids
    
    # Update z positions of the new splitted tracks
    # Use the fact, that ids with same z but different phi are next to each other
    new_z_positions = np.linspace(np.amin(bp_z_positions), np.amax(bp_z_positions), z_split, endpoint=False) + z_bin_size/2
    low_z_bounds = np.arange(0, phi_split*z_split, phi_split)
    high_z_bounds = low_z_bounds + phi_split
    assert len(new_z_positions) == len(low_z_bounds) == len(high_z_bounds) == z_split
    
    for low, high, new_pos in zip(low_z_bounds, high_z_bounds, new_z_positions):
        logging.debug("the ids %s get new z coord %f", np.arange(low, high), new_pos)
        prop_data.loc[ (prop_data['start_id'] >= low) & (prop_data['start_id'] < high), 'start_z' ] = new_pos
    
    if not return_z_distribution:
        return prop_data
    else:
        return prop_data, (z_ids, z_ids_weights)
    
    
#########
# TESTS #
#########

def test_uniform_beampipe_split(prop_data, bpz, bpphi):
    # First the 'old' method
    old_result = old_uniform_beampipe_split(prop_data.copy(), bpz, bpphi)
    
    # Then new method
    z_coords = prop_data.loc[ prop_data['start_id'] == 0]['pos_z'].to_numpy()
    z_split_bounds = make_uniform_z_split(z_coords, bpz)
    
    new_result = apply_beampipe_split(prop_data.copy(), z_split_bounds, bpphi)
    
    old_start_ids = old_result.loc[ old_result['start_id'] < bpz*bpphi, 'start_id' ].to_numpy().astype(int)
    new_start_ids = new_result.loc[ new_result['start_id'] < bpz*bpphi, 'start_id' ].to_numpy().astype(int)
        
    assert not np.absolute(new_start_ids - old_start_ids).any() > bpphi
    print("[ OK ] uniform beampipe split test passed")    
    
    
    
def visual_beampipe_split_test(prop_data):
    '''
    Test embeddings visually
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    z_split = 4
    phi_split = 1
    
    z_coords = prop_data.loc[ prop_data['start_id'] == 0]['pos_z'].to_numpy()
    split = make_uniform_z_split(z_coords, z_split)
    
    traf_data = apply_beampipe_split(prop_data.copy(), split, phi_split)
    traf_data_old = old_uniform_beampipe_split(prop_data.copy(), z_split, phi_split)
    
    #traf_data = traf_data_old
    
    colors = ['b', 'g', 'r', 'c']
    
    n_plot=0
    
    assert z_split*phi_split == 4
    
    for i in range(z_split*phi_split):
        data = traf_data.loc[ traf_data['start_id'] == i][['pos_x','pos_y','pos_z','dir_x','dir_y','dir_z']].to_numpy()
        idxs = np.random.randint(0,len(data),min(len(data), 50))
        
        surface_pos = traf_data.loc[ traf_data['start_id'] == i][['start_x','start_y','start_z']].to_numpy()[0]
        ax.scatter(surface_pos[0], surface_pos[1], surface_pos[2], c='black')
        
        for idx in idxs:
            p = data[idx][0:3]
            d = data[idx][3:6]
            pp = p + d
            ax.plot([ p[0], pp[0] ], [ p[1], pp[1] ], [ p[2], pp[2] ], colors[i])
            
            n_plot += 1
            
    
    logging.info("Plotted %d surfaces", n_plot)
        
    plt.show()


############################
# Test preprocessing chain #
############################

if __name__ == "__main__":    
    print("Test preprocessing chain...", flush=True)
    logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)
    
    root_dir = "/home/benjamin/Dokumente/acts_project/ml_navigator/data/"
    propagation_file = root_dir + "logger/navigation_training/data-201217-162645-n1.csv"
    detector_file = root_dir + "detector/detector_surfaces.csv"
    
    prop_data = pd.read_csv(propagation_file, dtype={'start_id': np.uint64, 'end_id': np.uint64})
    detector_data = pd.read_csv(detector_file, dtype={'geo_id': np.uint64})
    
    z_split = 4
    phi_split = 3
    
    
    visual_beampipe_split_test(prop_data)
    exit()
    
    def plot_real_space_embedding():
        z_split = 10
        phi_split = 5
        z_coords = prop_data.loc[ prop_data['start_id'] == 0]['pos_z'].to_numpy()
        split = make_uniform_z_split(z_coords, z_split)
        
        model = make_real_space_embedding_model(detector_data, split, phi_split)
        plot_embedding(model, detector_file, 3, z_split*phi_split)
        
        
    plot_real_space_embedding()
    exit()
    
    
    
    test_uniform_beampipe_split(prop_data, z_split, phi_split)
    
    ## Test alternative of custom beampipe split
    def test_custom_beampipe_split():
        z_coords = prop_data.loc[ prop_data['start_id'] == 0]['pos_z'].to_numpy()
        split = make_constant_density_z_split(z_coords, z_split)
        
        _, z_dist = apply_beampipe_split(prop_data.copy(), split, phi_split, return_z_distribution=True)
        
        print("[ OK ] test_custom_beampipe_split(...)", flush=True)
    
    test_custom_beampipe_split()
    
    
    ## Step 1: Beampipe split
    def test_uniform_beampipe_split():
        split = make_uniform_z_split(prop_data.loc[ prop_data['start_id'] == 0]['pos_z'].to_numpy(), z_split)
        step_1_res = apply_beampipe_split(prop_data.copy(), split, phi_split)

        # NOTE this only works at small split sizes, so all are bins are filled
        initial_num_unique_ids = len(np.unique(prop_data[['start_id','end_id']].to_numpy()))
        step_1_num_unique_ids = len(np.unique(step_1_res[['start_id','end_id']].to_numpy()))
        
        assert initial_num_unique_ids - 1 + z_split*phi_split == step_1_num_unique_ids
        
        print("[ OK ] beampipe_split(...)", flush=True)
        return step_1_res
    
    step_1_res = test_uniform_beampipe_split()
    
    
    ## Step 2: GeoID -> Number
    def test_geoid_to_ordinal_number():
        step_2_res = geoid_to_ordinal_number(step_1_res.copy(), detector_data, z_split*phi_split)
        step_2_unique_ids = np.unique(step_1_res[['start_id','end_id']].to_numpy())
        
        assert step_2_unique_ids.all() < len(step_2_unique_ids)
        
        print("[ OK ] geoid_to_ordinal_number(...)", flush=True)
        return step_2_res
    
    step_2_res = test_geoid_to_ordinal_number()
    
    
    ## Step 3: Categorize into tracks   
    def test_categorize_into_tracks():
        selected_params = ['dir_x','dir_y','dir_z']
        x_numbers, x_params, y_numbers = categorize_into_tracks(step_2_res.copy(), z_split*phi_split, selected_params)
        
        assert len(x_numbers) == len(x_params) == len(y_numbers)
        
        initial_sep_idxs = prop_data[prop_data['start_id'] == 0].index.to_numpy()
        track_lengths_ref = np.array([len(track) for track in x_numbers])
        track_lengths_check = np.append(np.diff(initial_sep_idxs), len(x_numbers[-1]))
        assert np.equal(track_lengths_ref, track_lengths_check).all()
        
        print("[ OK ] categorize_into_tracks(...)", flush=True)

    test_categorize_into_tracks()

