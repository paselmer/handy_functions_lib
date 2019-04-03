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



def make_video_from_images(video_name, img_file_list, fourcc, fps):
    """ This function takes a list of full-path image file names and 
        uses those images to stitch together a mp4 video frame-by-frame.
    """
    
    # INPUTS:
    # video_name    -> Full-path name of video file. Make extension ".avi" unless you know
    #                what you're doing.
    # img_file_list -> Text file containing list of full path image file names separated by
    #                  carriage returns.
    # fourcc        -> The FOURCC video code. Default is 0 to go with ".avi" file type.
    # fps           -> Integer frames per second.        

    frame = cv2.imread(img_file_list[0].strip())
    height, width, layers = frame.shape
    print(height,width,layers)

    # Define and create VideoWriter object
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in img_file_list:
        video.write(cv2.imread(image.strip()))

    video.release()
    #cv2.destroyAllWindows()
    
    return None
