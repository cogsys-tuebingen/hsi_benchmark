@echo off

if "%~1"=="" (
    echo Usage:
    echo download_data_set.bat ^<local folder^>
    exit /b 1
)

set "DATASET_FOLDER=%~1"

mkdir "%DATASET_FOLDER%" 2>nul

pushd "%DATASET_FOLDER%"

echo Download HRSS recordings from https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
mkdir "hrss_dataset"
cd hrss_dataset
powershell -command "Invoke-WebRequest -Uri https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat -OutFile Indian_pines_corrected.mat"
powershell -command "Invoke-WebRequest -Uri https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat -OutFile Indian_pines_gt.mat"
powershell -command "Invoke-WebRequest -Uri https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat -OutFile Salinas_corrected.mat"
powershell -command "Invoke-WebRequest -Uri https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat -OutFile Salinas_gt.mat"
powershell -command "Invoke-WebRequest -Uri https://www.ehu.eus/ccwintco/uploads/d/df/SalinasA.mat -OutFile SalinasA.mat"
powershell -command "Invoke-WebRequest -Uri https://www.ehu.eus/ccwintco/uploads/a/aa/SalinasA_gt.mat -OutFile SalinasA_gt.mat"
powershell -command "Invoke-WebRequest -Uri https://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat -OutFile Pavia.mat"
powershell -command "Invoke-WebRequest -Uri https://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat -OutFile Pavia_gt.mat"
powershell -command "Invoke-WebRequest -Uri https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat -OutFile PaviaU.mat"
powershell -command "Invoke-WebRequest -Uri https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat -OutFile PaviaU_gt.mat"
powershell -command "Invoke-WebRequest -Uri http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat -OutFile KSC.mat"
powershell -command "Invoke-WebRequest -Uri http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat -OutFile KSC_gt.mat"
powershell -command "Invoke-WebRequest -Uri http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat -OutFile Botswana.mat"
powershell -command "Invoke-WebRequest -Uri http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat -OutFile Botswana_gt.mat"
cd "%DATASET_FOLDER%"

echo Download DeepHS debris (resized) data set..
mkdir "deephs_debris_resized"
cd deephs_debris_resized
powershell -command "Invoke-WebRequest -Uri https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Debris-2023-Datasets/DeepHS-Debris-2023-Datasets.zip -OutFile debris.zip"
powershell -command "Expand-Archive -Path debris.zip -DestinationPath ."
move /Y DeepHS_debris\* .
rd /S /Q DeepHS_debris
del debris.zip
cd "%DATASET_FOLDER%"

echo Download DeepHS fruit (v4) data set..
mkdir "deephs_fruit_v4"
cd deephs_fruit_v4
powershell -command "Invoke-WebRequest -Uri https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Avocado.zip -OutFile Avocado.zip"
powershell -command "Expand-Archive -Path Avocado.zip -DestinationPath ."
del Avocado.zip
powershell -command "Invoke-WebRequest -Uri https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Kaki.zip -OutFile Kaki.zip"
powershell -command "Expand-Archive -Path Kaki.zip -DestinationPath ."
del Kaki.zip
powershell -command "Invoke-WebRequest -Uri https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Kiwi.zip -OutFile Kiwi.zip"
powershell -command "Expand-Archive -Path Kiwi.zip -DestinationPath ."
del Kiwi.zip
powershell -command "Invoke-WebRequest -Uri https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Mango.zip -OutFile Mango.zip"
powershell -command "Expand-Archive -Path Mango.zip -DestinationPath ."
del Mango.zip
powershell -command "Invoke-WebRequest -Uri https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Papaya.zip -OutFile Papaya.zip"
powershell -command "Expand-Archive -Path Papaya.zip -DestinationPath ."
del Papaya.zip
powershell -command "Invoke-WebRequest -Uri https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/annotations.zip -OutFile annotations.zip"
powershell -command "Expand-Archive -Path annotations.zip -DestinationPath ."
del annotations.zip

for /d %%i in (*) do (
    cd "%%i"
    ren "VIS" "SPECIM_FX10" 2>nul
    ren "VIS_COR" "CORNING_HSI" 2>nul
    ren "NIR" "INNOSPEC_REDEYE" 2>nul
    cd "%DATASET_FOLDER%\deephs_fruit_v4"
)

powershell -command "Invoke-WebRequest -Uri https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/readme.txt -OutFile readme.txt"
cd "%DATASET_FOLDER%"

popd
