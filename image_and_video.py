import os
import pdb
from file_and_os_routines import create_a_file_list
try:
    import cv2
except
    print("\n\nSome functions in this library require OpenCV (cv2).")
    print("It's been detected that you don't have this installed.")
    print("Until you install it, some functions might not work (you'll")
    print("get errors!).\n\n")


image_folder = 'C:\\Users\\pselmer\\Desktop\\Polarized_camera\\data\\Pol\\scan1_all_images_png\\'
video_name = 'scan1_video.avi'

# Create a file list
file_list = image_folder.split('\\')[-2]+'_file_list.txt'
search_str = '*.png'
create_a_file_list(file_list,image_folder,search_str)

# Read file names into a list object
with open(file_list) as f_obj:
	files = f_obj.readlines()

fnum = [] #file index number
ts = []   # timestamp
for file in files:
    substrings = file.split('\\')
    pathless_file_name = substrings[-1].split('_')
    fnum.append( int(pathless_file_name[2]) )
    ts.append( int(pathless_file_name[3].split('.')[0]) )

files_sorted = [x for _,x in sorted(zip(ts,files))]
fnum_sorted = [x for _,x in sorted(zip(ts,fnum))]
ts.sort() # sort in-place

def make_video_from_images():
    """ This function takes a list of full-path image file names and 
        uses those images to stitch together a mp4 video frame-by-frame.
    """

    frame = cv2.imread(files_sorted[0].strip())
    height, width, layers = frame.shape
    print(height,width,layers)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    video = cv2.VideoWriter(video_name, fourcc, 5.0, (width, height))

    for image in files_sorted:
        video.write(cv2.imread(image.strip()))

    video.release()
    cv2.destroyAllWindows()
