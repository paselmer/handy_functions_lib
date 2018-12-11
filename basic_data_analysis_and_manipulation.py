import numpy as np

def delineate_consecutive_binned_vector_values(arr,bins):
    """ Function will delineate bounds of consecutive runs of values in
        'arr' that fit within any of the given 'bins.'
        
        Return value change_indx will be an integer array containing the
        indicies at with the values change to a different bin. 
        Return value ncounts will be the number of consecutive values in
        arr in a bin.
    """
    
    # NOTES:
    # ui = the starting indicies of the consecutive value runs
    # ncounts = the count of each unique value run
    
    bin_numbers = np.digitize(arr,bins)
    dbn = bin_numbers[1:] - bin_numbers[:-1]
    change_bn_mask = dbn != 0
    indx = np.arange(arr.shape[0]-1)
    change_indx = indx[change_bn_mask] + 1
    # indexes where bin_numbers changes to a different bin
    change_indx = indx[change_bn_mask] + 1
    ncounts = np.zeros(change_indx.shape[0]+1,np.uint32)
    ncounts[0] = change_indx[0]
    ncounts[1:-1] = change_indx[1:] - change_indx[:-1]
    ncounts[-1] = arr.shape[0] - change_indx[-1]
    ui = np.zeros(change_indx.shape[0]+1,dtype=np.uint32)
    ui[1:] = change_indx
    
    return ui, ncounts
