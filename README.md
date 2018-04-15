# Chinese_Names_OCR

This repo aims to recognize names in traditional Chinese (MingLiu font).  
I used the CRNN model to do the following tasks.
- extract image features 
- explore sequential relationship in image features 
- transcript them to Chinese by CTC.

To train the model, I fixed image height to 36 pixel but keep the width flexible. The training image is in gray scacle.
I generated all images by using MingLiu font and fixed image size of 36 pixel in height and 248 pixel in width.



## Reference

@article{ShiBY15,
  author    = {Baoguang Shi and
               Xiang Bai and
               Cong Yao},
  title     = {An End-to-End Trainable Neural Network for Image-based Sequence Recognition
               and Its Application to Scene Text Recognition},
  journal   = {CoRR},
  volume    = {abs/1507.05717},
  year      = {2015}
}
