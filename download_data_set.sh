#!/bin/sh

if [ $# -eq 0 ]
  then
    echo "Usage:"
    echo "download_data_set.sh <local folder>"
    exit 1
fi

export DATASET_FOLDER="$1"

mkdir -p "$DATASET_FOLDER"

cd $DATASET_FOLDER

echo "Download HRSS recordings from https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes"
mkdir -p "$DATASET_FOLDER/hrss_dataset"
cd hrss_dataset
wget https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat
wget https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat
wget https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat
wget https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat
wget https://www.ehu.eus/ccwintco/uploads/d/df/SalinasA.mat
wget https://www.ehu.eus/ccwintco/uploads/a/aa/SalinasA_gt.mat
wget https://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat
wget https://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat
wget https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat
wget https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat
wget http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat
wget http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat
wget http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat
wget http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat
cd $DATASET_FOLDER

echo "Download DeepHS debris (resized) data set.."
mkdir -p "$DATASET_FOLDER/deephs_debris_resized"
cd deephs_debris_resized
wget -O debris.zip https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Debris-2023-Datasets/DeepHS-Debris-2023-Datasets.zip
unzip debris.zip 
mv DeepHS_debris/* .
rm -rf DeepHS_debris
rm debris.zip
echo $(pwd)
cd $DATASET_FOLDER

echo "Download DeepHS fruit (v4) data set.."
mkdir -p "$DATASET_FOLDER/deephs_fruit_v4"
cd deephs_fruit_v4
wget -O Avocado.zip https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Avocado.zip
unzip Avocado.zip 
rm Avocado.zip
wget -O Kaki.zip https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Kaki.zip
unzip Kaki.zip 
rm Kaki.zip
wget -O Kiwi.zip https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Kiwi.zip
unzip Kiwi.zip 
rm Kiwi.zip
wget -O Mango.zip https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Mango.zip
unzip Mango.zip 
rm Mango.zip
wget -O Papaya.zip https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Papaya.zip
unzip Papaya.zip 
rm Papaya.zip
wget -O annotations.zip https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/annotations.zip
unzip annotations.zip 
rm annotations.zip

# rename cameras for version 4
for i in $(ls .)
do
	cd $i
	mv VIS SPECIM_FX10 2> /dev/null
	mv VIS_COR CORNING_HSI 2> /dev/null
	mv NIR INNOSPEC_REDEYE 2> /dev/null
	cd "$DATASET_FOLDER/deephs_fruit_v4"
done

wget -O readme.txt https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/readme.txt
cd $DATASET_FOLDER

