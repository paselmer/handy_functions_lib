import numpy as np
from sklearn.metrics import auc
import pdb

def delineate_consecutive_binned_vector_values(arr,bins):
    """ Function will delineate bounds of consecutive runs of values in
        'arr' that fit within any of the given 'bins.'
        
        Return value change_indx will be an integer array containing the
        indicies at which the values change to a different bin. 
        Return value ncounts will be the number of consecutive values in
        arr in a bin.
    """
    
    # INPUT:
    # arr = a 1-D array (vector) containing the data values
    # bins = list or array defining the bounds of the bins
    
    # OUTPUTS:
    # ui = the starting indicies of the consecutive value runs
    # ncounts = the count of each unique value run
    # bin_numbers = the number of the bin you just entered into as corresponds
    #     to the above output vectors (ui, ncounts).
    
    bin_numbers = np.digitize(arr,bins)
    unique_vals = np.unique(bin_numbers)
    if unique_vals.shape[0] <= 1:
        print('All values of input are in a single bin. Returning Nones...')
        return None, None, None
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
    
    # Reduce bin_numbers so that it corresponds to other output vectors
    bn_out = np.zeros(ui.shape,dtype=bin_numbers.dtype)
    bn_out[-1] = bin_numbers[-1]
    bin_numbers = bin_numbers[:-1][change_bn_mask]
    bn_out[:-1] = bin_numbers
    bin_numbers = bn_out
    
    return ui, ncounts, bin_numbers
    
    
def clean_mean(x, mn, mx, min_N, null_val):
    """ Only use clean values to compute mean """
    mask = ( (x >= mn) & (x <= mx) & np.isfinite(x) )
    if mask.sum() >= min_N:
        result = np.mean( x[mask] )
    else:
        result = null_val
    return result

def smart_argmax(vector,w=5):
    """ If multiple argmaxs occur, pick the one with the least relief 
        (topographically-speaking) around it.
        vector => input 1D numpy array
        w => width centered on max points to search for 
    """
    mx = vector.max()
    mxmask = vector == mx
    n_mx = mxmask.sum()
    sargmax = np.nan # A bad invalid is returned if somehow it's not assigned.
    center_index = int(np.round(vector.shape[0] / 2))
    if n_mx == 1:
        #print('No need for smart_argmax()')
        return vector.argmax()
    hw = int(w/2)
   
    oords = np.arange(0,vector.shape[0]) 
    locs = oords[mxmask]
    # 1. If maxes are touching (adjacent),
    #    - If only 2, pick the one closest to center
    #    - If > 2, pick the one in the middle
    d = np.diff(locs)
    if (d == 1).sum() != 0: # A difference of 1 implies consecutive integers
        # --- Save non-consecutive peaks for analysis
        singular_peak_locs = []
        if d[0] != 1: singular_peak_locs.append(locs[0])
        for s in range(1,d.shape[0]):
            if ((d[s] > 1) and (d[s-1] != 1)): singular_peak_locs.append(locs[s])
        if d[-1] != 1: singular_peak_locs.append(locs[-1])
        # --- 
        if np.unique(d).shape[0] > 1: # Check to make sure all d's != 1.
            ic,nc,bn = delineate_consecutive_binned_vector_values(d,[0,1.1,2]) # bin 1 will be diff of 1
        else:  # Entering this block means that this is only 1 consecutive span
            ic = np.array([0])
            nc = np.array([locs.shape[0]-1]) # minus 1 because +1 will be added back [5/22/23]
            bn = np.array([1])
        diff_bin_mask = bn == 1
        if ic is None: pdb.set_trace()
        ic = ic[diff_bin_mask]
        nc = nc[diff_bin_mask] + 1
        middle_points = []
        for j in range(0,ic.shape[0]):
            print(j)
            start_index = locs[ic[j]]
            end_index = start_index + nc[j]
            print(end_index,start_index,end_index-start_index)
            print(vector[start_index:end_index])
            print(oords[start_index:end_index])
            x = oords[start_index:end_index]
            print(x.shape)
            # Now find the middle or suitable point
            if x.shape[0] == 2:
                middle_index = np.abs(x - center_index).argmin()
                middle_index = x[middle_index]
                print('middle index: ',middle_index)
                print('Shape of 2 block')
            else:
                middle_index = int(np.median(x))
            print(middle_index)
            print('---------------------')
            middle_points.append(middle_index)
        locs = np.array( middle_points + singular_peak_locs, dtype=int ) # pared consecutives and singulars back into modified locs variable
        n_mx = locs.shape[0] # New # of maxes, with consecutives cut out
    
    reliefs = np.full(n_mx,999)
    for i,argmax in enumerate(locs):
        reliefs[i] = (mx - vector[argmax-hw:argmax+hw+1]).sum()
        
    # 1. Choose max with least topographic relief
    mn_relief = reliefs.min()
    relief_mask = reliefs == mn_relief
    n_mnrf = relief_mask.sum()
    if n_mnrf == 1:
        i = locs[relief_mask]
        sargmax_val = vector[i]
        sargmax = i
    else: # Should have to be > 1
        # 2. Maxes are tied for topographic relief,
        #    choose the one closer to the center.
        imid = np.abs(locs[relief_mask] - center_index).argmin()
        i = locs[relief_mask][imid]
        sargmax_val = vector[i]
        sargmax = i
    #print('Standard argmax',vector.argmax(),vector[vector.argmax()])
    #print('Chosen argmax: ',sargmax,sargmax_val)
    if sargmax_val != vector.max(): pdb.set_trace()
    if sargmax_val != vector[sargmax]: pdb.set_trace()
    return int(sargmax)

def FWHM(X,Y):
    """ Compute approximate FWHM using discrete bins.
        In cases where FWHM is not definitionally met,
        a value is computed using boundaries.
    """
    HM = Y.max() / 2.0 # Half Max
    imax = smart_argmax(Y)
    right_edge = False
    left_edge = False
    if imax == 0:
        print('NOTE: Max of hist is at left edge')
        HMi_L = 0
        left_edge = True
    else:
        mask = (Y[:imax] <= HM)
        HMi_L = imax-np.flip(mask).argmax()-1  # index of HM on left
        if mask.sum() == 0: HMi_L=0 # No bin is lower than HM
    if imax >= Y.shape[0]-1:
        print('NOTE: Max of hist is at right edge')
        HMi_R = Y.shape[0] - 1
        right_edge = True
    else:
        mask = Y[imax+1:] <= HM
        HMi_R = (mask).argmax()+imax # index of HM on right
        if mask.sum() == 0: HMi_R = Y.shape[0] - 1
    try:
      FWHM = X[HMi_R] - X[HMi_L]
    except:
      pdb.set_trace()
    if FWHM == 0:
      print('NOTE: FWHM compuated as zero. \nArtificially making FWHM 1 bin unit long')
      if left_edge:
        HMi_R = 1
      elif right_edge: 
        HMi_L == HMi_R - 1
      else:
        HMi_R += 1
      FWHM = X[HMi_R] - X[HMi_L]
      
    #plt.close('all') # Debug plotting
    #plt.plot(X,Y)
    #plt.plot([X[HMi_L],X[HMi_R]],[HM,HM])
    #plt.plot([X[HMi_L],X[HMi_L]],[0,HM])
    #plt.plot([X[HMi_R],X[HMi_R]],[0,HM])
    #plt.show()
    #pdb.set_trace()
    return FWHM, X[HMi_R], X[HMi_L]

def compute_overlap_area(xp,yp1,yp2,dx):
    """ Given two curves with identical, evenly-spaced,
        monotomically-increasing  x-coordinates/points,
        compute the overlap area.
    """
    numpoints = xp.shape[0]
    overlapping_points = ((yp1 != 0) & (yp2 != 0))
    num_ovlp_points = overlapping_points.sum()
    ypo = np.zeros(numpoints) # y values of the overlap shape
    xoord = np.arange(0,numpoints)[overlapping_points] # xoordinate indexes
    runsum_area = 0.0
    for i in xoord: # This loop only iterates indexes of overlap
        ylow = min(yp1[i],yp2[i])
        runsum_area = runsum_area + (dx * ylow)
        ypo[i] = ylow
    return runsum_area, ypo

def intracluster_distributions_comparisons(d,labels,n_clusters,bins,tag,out_dir,filter_values=[]):
    """ Function to determine whether two distributions are substantially different
        AND where each distribution should continue to be split (via clustering).

        INPUTS:
        d -> data (data points, channels), prefferably in native units?
        labels -> labels for each data point (data points), values of either 0 or 1 denoting cluster
        n_clusters -> As of 4/25/23, assume this is 2
        histbins -> An array of histogram bins following numpy.histogram default convention
        tag -> ID tag for labeling plots and output file names, string
        out_dir -> directory for output (likely just plots)
  
        OUTPUTS:
        FWHM0 -> FWHM of cluster labeled as 0
        FWHM1 -> FWHM of cluster labeled as 1
        FWHM0xl -> x value (constrained by bins) of left edge of FWHM of cluster labeled as 0
        FWHM1xl -> x value (constrained by bins) of left edge of FWHM of cluster labeled as 1
        FWHM0xr -> x value (constrained by bins) of right edge of FWHM of cluster labeled as 0
        FWHM1xr -> x value (constrained by bins) of right edge of FWHM of cluster labeled as 1        
        overlap_area_n -> The amount of area overlap between both curves, computed after areas normalized to 1
        overlap_area_FWHM_n -> The overlap area within the leftmost and rightmost FWHM bounds, (normalized)
        overlap_area_FWHM_n_as_frac_of_larger -> expresses overlap_area_FWHM_n (defined above) as
            fraction of larger cluster's area within FWHM span.
    """
    d = np.squeeze(d)
    mask = np.ones(d.shape,dtype=bool)
    for value in filter_values:
        mask = (d != value) & mask
    d = d[ mask ]
    labels = labels[ mask ]
    midbin = (bins[1:] + bins[:-1]) / 2
    clust0 = labels == 0
    clust1 = labels == 1
    if ((mask.sum() == 0) or (clust0.sum()==0) or (clust1.sum() == 0)):
        print('\n All values filtered out!')
        return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1  
    #
    fig,ax = plt.subplots(1,2,figsize=(14,10))
    counts0,bins_,p_ = ax[0].hist(d[clust0],bins=bins,label='0', alpha=0.7)
    counts1,bins_,p_ = ax[0].hist(d[clust1],bins=bins,label='1', alpha=0.7)
    ax[0].legend()
    #
    # Compute AUC then normalize distribtuions so that both have AUC = 1
    auc0 = auc(midbin,counts0)
    auc1 = auc(midbin,counts1)
    print('AUC 0,1: {},{}'.format(auc0,auc1))
    counts0_n = counts0 / auc0
    counts1_n = counts1 / auc1
    auc0_n = auc(midbin,counts0_n)
    auc1_n = auc(midbin,counts1_n)
    print('AUC 0,1: {},{}'.format(auc0_n,auc1_n))
    bar_width = np.diff(bins).mean()
    ax[1].bar(midbin,counts0_n,alpha=0.7,width=bar_width)
    ax[1].bar(midbin,counts1_n,alpha=0.7,width=bar_width)
    # Weighted average depol value. Weighted according to frequency of occurence. 
    wavg0 = np.average(midbin,weights=counts0) # Should essentially be center of mass
    wavg1 = np.average(midbin,weights=counts1)
    avg0 = d[clust0].mean()
    avg1 = d[clust1].mean()
    print('weighted average, average : {},{} || {},{}'.format(wavg0,avg0,wavg1,avg1))
    print('Peaks 0, 1: {}, {}'.format(midbin[counts0.argmax()],midbin[counts1.argmax()]))
    # Full Width at Half Maximum (https://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak)
    FWHM0, FWHM0xr, FWHM0xl = FWHM(midbin,counts0)
    FWHM1, FWHM1xr, FWHM1xl = FWHM(midbin,counts1)
    print('FWHM clust0, clust1 {}, {}'.format(FWHM0,FWHM1))
    HM0 = counts0.max()/2
    HM1 = counts1.max()/2
    ax[0].plot([FWHM0xl,FWHM0xr],[HM0,HM0],color='red')
    ax[0].plot([FWHM1xl,FWHM1xr],[HM1,HM1],color='black')
    # Overlap area after normalization
    overlap_area_n, overlap_counts_n = compute_overlap_area(midbin,counts0_n,counts1_n,bar_width)
    print('Overlap area after normalizing area = {}/1.0'.format(overlap_area_n))
    ax[1].bar(midbin,overlap_counts_n,alpha=1.0,color='green',width=bar_width)
    # Compute AUC between FWHM x-oords (non-normed area)
    xil0 = np.abs(midbin - FWHM0xl).argmin()
    xir0 = np.abs(midbin - FWHM0xr).argmin()
    auc_FWHM0 = auc(midbin[xil0:xir0+1],counts0[xil0:xir0+1])
    xil1 = np.abs(midbin - FWHM1xl).argmin()
    xir1 = np.abs(midbin - FWHM1xr).argmin()
    auc_FWHM1 = auc(midbin[xil1:xir1+1],counts1[xil1:xir1+1])    
    # Compute fraction of area within the FWHM bounds
    frac_area_FWHM0 = auc_FWHM0 / auc0
    frac_area_FWHM1 = auc_FWHM1 / auc1
    print('Fraction area within FWHM 0,1 {}, {}'.format(frac_area_FWHM0,frac_area_FWHM1))
    # Within the FWHM bounds, what's the areal overlap using normalized areas
    lefti = min(xil0,xil1)
    righti = max(xir0,xir1)
    overlap_area_FWHM_n, _ = compute_overlap_area(midbin[lefti:righti+1],counts0_n[lefti:righti+1],counts1_n[lefti:righti+1],bar_width)
    ax[1].bar(midbin[lefti:righti+1],_,alpha=1.0,color='black',width=bar_width)
    # Now report FWHM areal overlap as fraction of bigger curve's area
    # In other words, this % of bigger curve is overlapped by smaller curve
    auc0_overspan = auc(midbin[lefti:righti+1],counts0_n[lefti:righti+1])
    auc1_overspan = auc(midbin[lefti:righti+1],counts1_n[lefti:righti+1])
    overlap_area_FWHM_n_as_frac_of_larger = overlap_area_FWHM_n/max(auc0_overspan,auc1_overspan)
    print('Overlap area within FWHM using normalized area = {}'.format(overlap_area_FWHM_n))
    print("Overlap area within FWHM as frac of larger curve's area = {}".format(overlap_area_FWHM_n_as_frac_of_larger))
    plt.savefig(out_dir + tag + '_distributions_summary.png')
    #pdb.set_trace()
    #pdb.set_trace()
    #plt.show()
    plt.close()
    return FWHM0, FWHM1, FWHM0xl, FWHM1xl, FWHM0xr, FWHM1xr, overlap_area_n, overlap_area_FWHM_n, overlap_area_FWHM_n_as_frac_of_larger, frac_area_FWHM0, frac_area_FWHM1
