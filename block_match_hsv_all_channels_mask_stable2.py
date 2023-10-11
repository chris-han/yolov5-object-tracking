import cv2,math,time
import numpy as np
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from sklearn.cluster import DBSCAN
# import skvideo.io
# from PIL import Image

# Define the number of frames to use for the moving average filter
num_frames_moving_average = 5
num_consecutive_still_frames = 15
# size of matching block for scene change detection
block_size = 32

# if the mse is above a threshold, mark it as a changed block
mse_threshold = 50
# if mse_v2h_ratio is below a threshold, mark it as a none-shadow changed block
mse_v2h_threshold = 0.6
# frame size for uploading to cloud
frame_size = 640
half_frame_size = int(frame_size/2)

# frame_movement_tolerance: the higher the value, the less sensitive to frame movement
frame_movement_tolerance = int(frame_size*0.6)

previous_center_x = half_frame_size
previous_center_y = half_frame_size
# Open the video file
video_file_path = 'a.mp4'
cap = cv2.VideoCapture(video_file_path)

# Create a buffer to store the previous frames
frame_buffer_original = []
frame_buffer_hsv = []
# create a np array smooth_frame_buffer to store the smoothed frames
frame_buffer_smoonth = []

# Create a video writer object with mjpeg codec
codec = cv2.VideoWriter_fourcc(*'avc1')


# mask_data = np.load('a_mask.npy')

#load png file to np array
mask_data = cv2.imread('a_mask_gray.png', cv2.IMREAD_GRAYSCALE)
# alpha_image = Image.fromarray(mask_data, mode='L').convert('LA')
# # Save the image as a PNG file
# alpha_image.save('a_mask_gray.png')



# extract the cordinates of the four corners of the image in alpha_data
# find the min and max of the x and y cordinates, y first and then x out of np.where
y,x = np.where(mask_data==255)
x_min, x_max = min(x), max(x)
y_min, y_max = min(y), max(y)
# adjust the x_min, x_max, y_min, y_max to make sure the difference is not less then frame_size
mask_width = x_max-x_min
if mask_width < frame_size:
    x_min = x_min - int((frame_size-mask_width)/2)
    x_max = x_max + int((frame_size-mask_width)/2)
    if x_max > 1920:
       x_min = 1920 - frame_size
       x_max = 1920

mask_height = y_max-y_min
if mask_height < frame_size:
    y_min = y_min - int((frame_size-mask_height)/2)
    y_max = y_max + int((frame_size-mask_height)/2)
    if y_max > 1080:
       y_min = 1080 - frame_size
       y_max = 1080
    


# fps = cap.get(cv2.CAP_PROP_FPS)

# Get the video metadata using ffprobe
# metadata = skvideo.io.ffprobe(video_file_path)
# Extract the frame rate from the metadata
# Split the fraction into numerator and denominator
# numerator, denominator = metadata['video']['@avg_frame_rate'].split('/')
# # Convert the numerator and denominator to float numbers
# numerator = float(numerator)
# denominator = float(denominator)
# # Calculate the frame rate as the numerator divided by the denominator
# fps = numerator / denominator

# fps = metadata['video']['@avg_frame_rate']
# prefix 'cropped_' to the video_file_path
# output_video_path = 'cropped_' + video_file_path
# output_video = skvideo.io.FFmpegWriter(output_video_path, outputdict={ '-vcodec': 'libx264', "-pix_fmt": "yuv444p", '-r': '{}'.format(fps), '-s': '{}x{}'.format(frame_size, frame_size)})
# Iterate over each frame in the video
while cap.isOpened():
    # Read the next frame from the video
    ret, frame_src = cap.read()

    # If the frame was successfully read, apply the filter
    if ret:
        # mask the frame with alpha layer
        frame_src = cv2.bitwise_and(frame_src, frame_src, mask=mask_data)
        # crop the frame with x_min, x_max, y_min, y_max
        frame_src = frame_src[y_min:y_max, x_min:x_max]

        # initialize frame_buffer_original with frame and 0 indicating no change
        frame_buffer_original.append([frame_src,0])        
        # save frame to file
        # cv2.imwrite('a_masked.png', frame)

        # Convert the frame to HSV format
        hsv_frame = cv2.cvtColor(frame_src, cv2.COLOR_BGR2HSV)
        # Add the current frame to the buffer
        frame_buffer_hsv.append(hsv_frame)
        # Split the image into its component channels
        #hsv_channels = cv2.split(hsv_frame)
        #cv2.imshow('Hue Channel', hsv_channels[0])
        #cv2.imshow('Saturation Channel', hsv_channels[1])
        #cv2.imshow('Value Channel', hsv_channels[2])

        
        # If the buffer is full, apply the moving average filter
        if len(frame_buffer_hsv) == num_frames_moving_average:
            # Compute the average of the frames in the buffer
            avg_frame = np.mean(frame_buffer_hsv, axis=0).astype(np.uint8)

            # Convert the output frame back to BGR format for display
            smooth_frame = cv2.cvtColor(avg_frame, cv2.COLOR_HSV2BGR)
            # add out_frame to smooth_frame_buffer
            frame_buffer_smoonth.append(smooth_frame)

            # Display the smoothed output frame
            cv2.imshow('Smoothed Frame', smooth_frame)

            # Remove the oldest frame from the buffer
            frame_buffer_hsv.pop(0)

        # Wait for a key press and exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


# destroy window named Smoothed Frame
cv2.destroyWindow('Smoothed Frame')

# initialize previous_frame with the first frame from smooth_frame_buffer
previous_frame = frame_buffer_smoonth[0]
# Convert the frame to the HSV color space
previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2HSV)

# Convert the frame to grayscale
current_frame_g = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2HSV)
# Get the size of the frame
rows, cols = current_frame_g.shape[:2]

#previous_frame = denoise_wavelet(previous_frame, channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)
overall_upper_left = (0,0)
overall_bottom_right = (cols,rows)

# remove the first num_buffer_frames-1 frames from frame_buffer_original to make it align with frame_buffer_smoonth
frame_buffer_original = frame_buffer_original[num_frames_moving_average-1:]
# print(len(frame_buffer_original)-len(frame_buffer_smoonth))

# interage over each frame in smooth_frame_buffer
for f in range(1, len(frame_buffer_smoonth)):
    # Read the next frame
    current_frame = frame_buffer_smoonth[f]
    # read the next frame from frame_buffer_src
    current_frame_original_color = frame_buffer_original[f][0]
        
    # initialize motion_blocks to empty array
    motion_blocks = np.empty((0, 2), int)

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):            

            current_block = current_frame[i:i+block_size, j:j+block_size, :]
            previous_block = previous_frame[i:i+block_size, j:j+block_size, :]
            
            # Compute the mean squared error between the blocks on color not grayscale
            mse = np.mean((previous_block - current_block) ** 2)  
            #print(mse)
            # Compute the mean squared error between the block and the original image
            mse_hue = np.mean((previous_block[:, :, 0] - current_block[:, :, 0]) ** 2)
            mse_sat = np.mean((previous_block[:, :, 1] - current_block[:, :, 1]) ** 2)
            mse_val = np.mean((previous_block[:, :, 2] - current_block[:, :, 2]) ** 2)
            # Print the mean squared error for each channel            

            

            # If the mse is above a threshold, mark it as a changed block
            if mse > mse_threshold:
                mse_v2h_ratio = mse_val/mse_hue
                # mse_v2s_ratio = mse_val/mse_sat                
                if mse_v2h_ratio <mse_v2h_threshold:                    
                    j2=j+block_size
                    i2=i+block_size

                    # draw a green rectangle around the changed block
                    # cv2.rectangle(current_frame_original_color, (j, i), (j2,i2), (0, 255, 0), 3)                    
                    # draw mse_hue, mse_sat, mse_val on the frame
                    cv2.putText(current_frame_original_color, '{:.2f}'.format(mse_v2h_ratio), (j, i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                    print('mse_v2h_ratio = {:.2f}'.format(mse_v2h_ratio))
                    
                    motion_blocks = np.append(motion_blocks, [(j, i)], axis=0)
                    
                    # only weight the last 10 changed blocks for clustering
                    motion_blocks = motion_blocks[-num_frames_moving_average:]
                    # maximum distance between two points to be considered as part of the same cluster(eps) = 100
                    clustering = DBSCAN(eps=100, min_samples=3).fit(motion_blocks)
                    labels = clustering.labels_
                    # only get the labels that are not outliers
                    # labels = labels[labels >= 0]
                    for label in np.unique(labels):
                        # ignore the outliers which are labeled as -1
                        if label == -1:
                            continue
                        hot_area =motion_blocks[labels == label]
                        # print("label:", label)
                        # print(hot_area)
                        # draw a red rectangle around the overall changed area
                        center_x = int(np.mean(hot_area[:, 0]))
                        center_y = int(np.mean(hot_area[:, 1]))

                        if abs(center_x-previous_center_x) > frame_movement_tolerance or abs(center_y-previous_center_y) > frame_movement_tolerance:                           
                            previous_center_x = center_x
                            previous_center_y = center_y
                        else:
                            center_x = previous_center_x
                            center_y = previous_center_y

                        # compute median of x and y
                        # center_x = int(np.median(hot_area[:, 0]))
                        # center_y = int(np.median(hot_area[:, 1]))


                        # keep the center of the overall changed area within the frame
                        if center_x < half_frame_size: 
                            upper_left_x = 0 
                        elif center_x>cols-half_frame_size:
                            center_x = cols-half_frame_size 
                        else: 
                            upper_left_x = center_x-half_frame_size

                        if center_y < half_frame_size:
                            upper_left_y = 0
                        elif center_y>rows-half_frame_size:
                            center_y = rows-half_frame_size
                        else:
                            upper_left_y = center_y-half_frame_size

                        upper_left_x = center_x-half_frame_size if center_x>=half_frame_size else 0
                        upper_left_y = center_y-half_frame_size if center_y>=half_frame_size else 0
                        bottom_right_x = frame_size+upper_left_x
                        bottom_right_y = frame_size+upper_left_y
                        
                        overall_upper_left = (upper_left_x, upper_left_y)
                        overall_bottom_right = (bottom_right_x, bottom_right_y)

    # crop the overall changed area from current_frame_original_color
    img_cropped = current_frame_original_color[overall_upper_left[1]:overall_bottom_right[1], overall_upper_left[0]:overall_bottom_right[0]]
    frame_buffer_original[f][0] = img_cropped
    # set frame change marker to 1 if motion_blocks is empty, meaning no change
    if motion_blocks.size == 0:
        frame_buffer_original[f][1] = 1
        # cv2.line(img_cropped, (0, 0), (frame_size, frame_size), (0, 0, 255), 5)

    cv2.imshow('Cropped Frame', img_cropped)
    # output the frame to a video file 'cropped_output.mp4' with out
    # out.write(crop_img)



    # Display the result    
    #cv2.rectangle(current_frame_original_color, overall_upper_left, overall_bottom_right, (0, 0, 255), 5)  
    # cv2.imshow('Block Matching', current_frame_original_color)    
    # print("-------------------------")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # The current frame becomes the previous frame for the next iteration
    previous_frame = current_frame.copy()

# if frame_buffer_original[f][1] has 25 consecutive 1s, then set all frame_buffer_original[f][1] to 1 otherwise 0
for f in range(1, len(frame_buffer_original)-num_consecutive_still_frames):
    if frame_buffer_original[f][1] == 1:
        for i in range(0, num_consecutive_still_frames-1):
            if frame_buffer_original[f+i][1] == 1:
                frame_buffer_original[f][1] = 1
            else:
                frame_buffer_original[f][1] = 0
                break
        if frame_buffer_original[f][1] == 1:
            for i in range(0, num_consecutive_still_frames-1):
                # draw a red cross on the frame
                cv2.line(frame_buffer_original[f+i][0], (0, 0), (frame_size, frame_size), (0, 0, 255), 5)

                # fill frame_buffer_original[f+i][0] with black
                # frame_buffer_original[f+i][0] = np.zeros((frame_size,frame_size,3), np.uint8)

            f = f+num_consecutive_still_frames-1
           
    
for f in range(1, len(frame_buffer_original)):
    # convert crop_img to skimage format
    frame_origi = frame_buffer_original[f][0] 
    frame_origi = cv2.cvtColor(frame_origi, cv2.COLOR_BGR2RGB)
    # convert crop_img to np array and write to output_video    
    frame_out = np.asarray(frame_origi, dtype=np.uint8)
    # output_video.writeFrame(frame_out)

# Release the video file and close all windows
cap.release()
# out.release()
# output_video.close()
cv2.destroyAllWindows()
