wget https://roozbehm.info/pascal-parts/trainval.tar.gz
mkdir data/pascal_part/
tar -xvf trainval.tar.gz -C data/pascal_part/
rm trainval.tar.gz

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
mkdir data/pascal_part/images
tar -xvf VOCtrainval_11-May-2012.tar -C data/pascal_part/images --strip-components=3 VOCdevkit/VOC2012/JPEGImages/
rm VOCtrainval_11-May-2012.tar
