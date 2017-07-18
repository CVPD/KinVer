KinfaceW-II Dataset

Overview:


The KinfaceW-II Dataset contains the parents-children images used in the paper "Neighborhood Repulsed Metric Learning for Kinship Verification" (CVPR 2012) by Jiwen Lu, Junlin Hu, Xiuzhuang Zhou, Yuanyuan Shang, Yap-Peng Tan and Gang Wang.

There are four kinship relations in the KinFaceW-II dataset: Father-Son (FS), Father-Daughter (FD), Mother-Son (MS), and Mother-Daughter (MD). In the KinFaceW-II dataset, each relationship contains 250 pairs of kinship images.

In this dataset, there are four folders: father-dau, father-son, mother-dau and mother-son, representing four different kinship relations: Father-Daughter (FD), Father-Son (FS), Mother-Daughter (MD), and Mother-Son (MS). The file name in the father-dau folder is fd_xxx_a, here xxx indicates the xxxth person, a=1 indicates the parent, and a=2 indicates the child.  

For ease of use, we manually label the coordinates of the eyes position of each face image, and cropped and aligned facial region into 64x64.




Meta data:

For each relation, we split images as 5-fold, four folds are used for traing model and other one for testing, and the mean of these five folds is recorded as the final result. 

In meta_data folder, the structure of the *.mat file is as follows:
fold   kin/non-kin   image1   image2


For KinFaceW-II dataset, each *.mat file lists 250 positive pairs and 250 negative pairs of images. And for each fold, there are 50 positive and 50 negative pairs, and there are no overlopping persons for all the folds. 




Terms of Use:

Please adhere to the following terms of use of this dataset. 

This dataset is for non-commercial reseach purposes (such as academic research) only. The images are not allowed to be redistributed (do not pass copies of any part of this collection to others, or post any images on the Internet). 

If you use any part of this image collection in your research, please cite the paper below.

Citation
Jiwen Lu, Junlin Hu, Xiuzhuang Zhou, Yuanyuan Shang, Yap-Peng Tan and Gang Wang. Neighborhood Repulsed Metric Learning for Kinship Verification, IEEE International Conference on Computer Vision and Pattern Recognition (CVPR'12), 2012.

Bibtex
@inproceedings{lu2012neighborhood,
  title={Neighborhood repulsed metric learning for kinship verification},
  author={Lu, J. and Hu, J. and Zhou, X. and Shang, Y. and Tan, Y.-P. and Wang, G.},
  booktitle={IEEE International Conference on Computer Vision and Pattern Recognition, 2012},
  pages={2594--2601},
  year={2012},
  organization={IEEE}
}

Please contact Jiwen Lu at jiwen.lu@adsc.com.sg or elujiwen@gmail.com if there is any problem on using the datasets.

Please inform us about your accuracies on these datasets, so that we may make them publicly available for easy comparison and citation.