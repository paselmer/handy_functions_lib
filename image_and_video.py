import os
import pdb
from file_and_os_routines import create_a_file_list
try:
    import cv2
except:
    print("\n\nSome functions in this library require OpenCV (cv2).")
    print("It's been detected that you don't have this installed.")
    print("Until you install it, some functions might not work (you'll")
    print("get errors!).\n\n")


image_folder = '/cpl3/PELICANS/Genie_Nano_M2450/outdoor_skyview_29mar19/PNG_images/'
video_name = 'sky_video_8fps.avi'
fps = 8
delim = '/'

# Create a file list
file_list = image_folder.split(delim)[-2]+'_file_list.txt'
search_str = '*.png'
create_a_file_list(file_list,image_folder,search_str)

# Read file names into a list object
with open(file_list) as f_obj:
	files = f_obj.readlines()

fnum = [] #file index number
ts = []   # timestamp
for file in files:
    substrings = file.split(delim)
    pathless_file_name = substrings[-1].split('_')
    fnum.append( int(pathless_file_name[-2]) )
    ts.append( int(pathless_file_name[-1].split('.')[0]) )

files_sorted = [x for _,x in sorted(zip(ts,files))]
fnum_sorted = [x for _,x in sorted(zip(ts,fnum))]
ts.sort() # sort in-place

def make_video_from_images(video_name, img_file_list, fourcc=0, fps):
    """ This function takes a list of full-path image file names and 
        uses those images to stitch together a mp4 video frame-by-frame.
    """
    
    # INPUTS:
    # video_name -> Full-path name of video file. Make extension ".avi" unless you know
    #                what you're doing.
    # img_file_list -> Text file containing list of full path image file names separated by
    #                  carriage returns.
    # fourcc -> The FOURCC video code. Default is 0 to go with ".avi" file type.
    # fps -> Integer frames per second.        

    frame = cv2.imread(files_sorted[0].strip())
    height, width, layers = frame.shape
    print(height,width,layers)

    # Define and create VideoWriter object
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in files_sorted:
        video.write(cv2.imread(image.strip()))

    video.release()
    #cv2.destroyAllWindows()
    
    return None
