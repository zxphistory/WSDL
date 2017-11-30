# WSDL
This ia an implementation of paper 
`Weak to Strong Detector Learning for Simultaneous Classification and Localization'

installation
1. Compile matconvnet toolbox using cnn_wsddn_compilenn: (see the options in http://www.vlfeat.org/matconvnet/install/

2. Download the PASCAL VOC 2007 devkit and dataset http://host.robots.ox.ac.uk/pascal/VOC/ under data folder

3. Download the pre-computed edge-boxes from the links below (for trainval and test splits):

    https://drive.google.com/file/d/1nJ0WOSNe3rI63MnRB3-_wtinLHESmayT/view?usp=sharing
    https://drive.google.com/file/d/14hRCsiAzt3eTyOTw3madR_L5rlMC4YqK/view?usp=sharing

Train and test models

Run scripts/Main_Script.m 

Demo

Run scripts/Demo.m

For demo evaluation, pretrained features based on caffenet and vgg-vd is available 

caffenet   https://drive.google.com/file/d/16WLv1zPTNSE9iOdmHD5AjrSmCJDtOfSc/view?usp=sharing

vgg-vd     https://drive.google.com/file/d/16AqNvfbTJ0lEzHOqoqepyydKakXew8IM/view?usp=sharing 

imdb_eval  https://drive.google.com/file/d/1SCnAIbsAKfKE4o22cCkoEXN_qvzFt7gC/view?usp=sharing


License

The analysis work performed with the program(s) must be non-proprietary work. Licensee and its contract users must be or be affiliated with an academic facility. Licensee may additionally permit individuals who are students at such academic facility to access and use the program(s). Such students will be considered contract users of licensee. The program(s) may not be used for commercial competitive analysis (such as benchmarking) or for any commercial activity, including consulting.
