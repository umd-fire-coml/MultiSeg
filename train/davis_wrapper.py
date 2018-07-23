# DAVIS dataset API

##################################################################
#
# Usage:
# dataset = DavisDataset("DAVIS", "480p") 
#
##################################################################

import numpy as np
import skimage.io as io
import glob
from sklearn.model_selection import train_test_split

class DavisDataset(object):
  
    # TODO: in init include way to download dataset
    # include download link and expected directory structure
  
    def __init__(self, directory, quality, val_videos=[]):
        
        """
        self.frame_pairs = an array of tuples of the form:
        (img_prev, img_curr, mask_prev, mask_curr) PATHS
        
        """
        
        # generate mask pairs
        
        self.trn_frame_pairs = [] # tuples of image and masks at t-1 and t
        self.val_frame_pairs = []
        
        image_dir = "%s/JPEGImages/%s/" % (directory, quality)
        mask_dir = "%s/Annotations/%s/" % (directory, quality)
        
        ### CHANGE HERE: splitting into train_videos and val_videos
        videos = [x[len(image_dir):] for x in glob.glob(image_dir + "*")]
        self.trn_videos = list(set(videos) - set(val_videos))
        self.val_videos = val_videos
        
        ### CHANGE HERE:
        for video in videos:         
      
            frames = [x[len(image_dir) + len(video) + 1:-4] for x in glob.glob(image_dir + video + "/*")]
            frames.sort()
      
            for prev, curr in zip(frames[:-1], frames[1:]):
        
                image_prev = image_dir + video + "/" + prev + ".jpg"
                image_curr = image_dir + video + "/" + curr + ".jpg"
                mask_prev = mask_dir + video + "/" + prev + ".png"
                mask_curr = mask_dir + video + "/" + curr + ".png"
                
                if video in self.trn_videos:
                    self.trn_frame_pairs.append( (image_prev, image_curr, mask_prev, mask_curr) )
                else:
                    self.val_frame_pairs.append( (image_prev, image_curr, mask_prev, mask_curr) )
  
    def get_train_val(self, shuffle=True, random_state=42):
        if shuffle:
            np.random.seed(random_state)
            trn_frame_pairs = self.trn_frame_pairs.copy()
            val_frame_pairs = self.val_frame_pairs.copy()
            np.random.shuffle(trn_frame_pairs)
            np.random.shuffle(val_frame_pairs)
            return trn_frame_pairs, val_frame_pairs
        return self.trn_frame_pairs, self.val_frame_pairs
    
    def get_video_split(self):
        return self.trn_videos, self.val_videos
    
    def get_random_pair(self, val=True, random_state=42):
        # returns from training
        np.random.seed(random_state)
        if val:
            return self.val_frame_pairs[np.random.choice(range(len(self.val_frame_pairs)))]
        return self.trn_frame_pairs[np.random.choice(range(len(self.trn_frame_pairs)))]
    
    def data_generator(self, frame_pairs, get_model_input, batch_size=4, shuffle=True, random_seed=42):
        
        """
        Arguments:
        :param frame_pairs
        :param get_model_input -> Function to get model input given a frame pair (i.e. apply Optical Flow)
        """
    
        np.random.seed(random_seed)
        np.random.shuffle(frame_pairs)
    
        i = 0
    
        while True:
        
            batch_count = 0
            X = []
            y = []
        
            while batch_count < batch_size:
            
                sample = frame_pairs[i]
                model_input, ground_truth = get_model_input(*sample)
            
                if model_input is not None and ground_truth is not None:
                    X.append(model_input)
                    # THIS PART IS WRONG: makes the output dimensions (batch_size, 480, 864, 1)
                    # But I don't think it impacts the output
                    y.append(np.expand_dims(ground_truth, axis=3))
                    batch_count = batch_count + 1
            
                i = i + 1 # keeps track of where we are in dataset
                
                # restart -> may cause repeats in samples within epoch or validation run
                if i >= len(frame_pairs):
                    i = 0 # go back to beginning
                    np.random.shuffle(frame_pairs)
            
            X = np.array(X)
            y = np.array(y)
        
            yield X, y