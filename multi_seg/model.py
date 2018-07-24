import os

############################################################
# Optical Flow (PWC Net)
############################################################

############################################################
# Image Segmentation (Mask RCNN)
############################################################

############################################################
# Mask Propagation
############################################################

############################################################
# Instance ID
############################################################

############################################################
# Mask Fusion
############################################################

############################################################
# MultiSeg Model
############################################################



#Boot Strap
#Image t is put through Image Segmentation, and Masks and ID info are stored
############################################################



#With Previous Mask

#Images t-1, t are put through optical flow, output is recieved as two channel image

#t-1 Masks and ID, as well as output from optical flow are put through Mask Propogation to
#       make predictive masks for all previous IDs

#In parallel, Image t is put through Image Segmentation Masks and ID info are stored

#The predictive masks, previous ID and featuremap, and the masks for the current image are
#       run through Instance ID to determine new masks ID

#If there are masks with no matching ID (t-1) new mask is used
#If there are masks with no matching ID (t) predictive mask is used in place
#If there are matching masks, both the new mask and predictive mask will be
#       fused through the Mask Fusion module where a refined mask will be created

#All masks will be compiled into a mask layer and placed on a frame in the video