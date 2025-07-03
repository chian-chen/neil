INTRODUCTION
------------------

Project page: https://github.com/kaifuyang/Gray-Pixel

NCC dataset is a novel dataset of Nighttime Color Constancy that contains 513 nighttime images and corresponding ground-truth illuminants. All images were taken by the Nikon D750 from nighttime outdoor scenes. These images are collected in two periods with different ISO settings, 6400 for the first 311 images (#1-#311 in the dataset) and 500 for the last 202 images (#312-#513 in the dataset).


CONTENTS
------------------
img        --  513 original linear images (.png)
msk        --  color checker masks
gt.mat    --  groudtruth illuminants of all the images
imlist.txt --  all the image names


CITATION
------------------
If you use NCC dataset, please cite:

Cheng Cheng, Kai-Fu Yang, Xue-Mei Wan, Leanne Lai Hang Chan, and Yong-Jie Li, "Nighttime color constancy using robust gray pixels," J. Opt. Soc. Am. A. 41(3), 476-488 (2024)

@article{cheng2024nighttime,
  title={Nighttime Color Constancy Using Robust Gray Pixels},
  author={Cheng, Cheng and Yang, Kai-Fu and Wan, Xue-Mei and Chan, Leanne Lai Hang and Li, Yong-Jie},
  journal={Journal of the Optical Society of America A},
  volume={41},
  number={3},
  pages={476--488},
  year={2024},
  publisher={Optica Publishing Group}
}