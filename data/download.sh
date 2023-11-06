#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # no color
DIR="$(dirname $0)"
ARGS=${@: 2};

# Download and decompress datasets

Help()
{
   # Display Help
   echo 
   echo "We provides four datasets to evaluate JOIN queries over unstructured data:"
   echo
   echo "Syntax: download/remove [dataset_name]"
   echo "options:"
   echo "help                         Print this Help."
   echo "download                     Download dataset"
   echo "remove                       Remove dataset"

   echo
   echo "======= Available datasets ======="
   echo "twitter                      twitter dataset (about 3.0MB)"
   echo "quora                        quora dataset (about 710KB)"
   echo "company                      company dataset (about 137MB)"
   echo "city_vehicle                 AICity Vehicle multi-camera multi-target tracking dataset (about 469MB)"
   echo "city_vehicle_2               AICity Vehicle multi-camera multi-target tracking dataset two table version (about 336MB)"
}

twitter()
{
    if [ ! -d "${DIR}/twitter/" ];
    then
        echo "Downloading twitter dataset(about 3.0MB)..."
        gdown 1VY3wFFRaRqR1HqwmTkPHwWb-DZpocFce

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/twitter.tar -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/twitter.tar

        echo -e "${GREEN}twitter dataset downloaded!${NC}"
    else
        echo -e "${RED}twitter dataset already exists under ${DIR}/twitter/!"
fi
}

quora()
{
    if [ ! -d "${DIR}/quora/" ];
    then
        echo "Downloading quora dataset(about 720KB)..."
        gdown 1ztcRe6iIbGrwCuWC4p_Pfwu1R832ntAy

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/quora.tar -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/quora.tar

        echo -e "${GREEN}quora dataset downloaded!${NC}"
    else
        echo -e "${RED}quora dataset already exists under ${DIR}/quora/!"
fi
}

company()
{
    if [ ! -d "${DIR}/company/" ];
    then
        echo "Downloading company dataset(about 137MB)..."
        gdown 13WMkYPr4olnlm8Vok4hs-1OuuiCwW1lD

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/company.tar -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/company.tar

        echo -e "${GREEN}company dataset downloaded!${NC}"
    else
        echo -e "${RED}company dataset already exists under ${DIR}/company/!"
fi
}

city_vehicle()
{
    if [ ! -d "${DIR}/city_vehicle/" ];
    then
        echo "Downloading AICity vehicle multi-camera multi-target tracking dataset(about 469MB)..."
        gdown 1pC2tltQrZT7f7gWI47wWL4sSjwMwx-Z9

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/city_vehicle.tar -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/city_vehicle.tar

        echo -e "${GREEN}AICity vehicle multi-camera multi-target tracking dataset downloaded!${NC}"
    else
        echo -e "${RED}AICity vehicle multi-camera multi-target tracking dataset already exists under ${DIR}/city_vehicle/!"
fi
}

city_vehicle_2()
{
    if [ ! -d "${DIR}/city_vehicle_2/" ];
    then
        echo "Downloading AICity vehicle multi-camera multi-target tracking dataset 2 tables (about 336MB)..."
        gdown 1XXaffI1GqThu3ucvpMRJfpZrg22gTwxB

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/city_vehicle_2.tar -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/city_vehicle_2.tar

        echo -e "${GREEN}AICity vehicle multi-camera multi-target tracking dataset downloaded!${NC}"
    else
        echo -e "${RED}AICity vehicle multi-camera multi-target tracking dataset already exists under ${DIR}/city_vehicle_2/!"
fi
}


Download() {
    for data in $ARGS
    do  
        echo "Downloading ${data}"
        case $data in
            twitter )
                twitter
                ;;
            quora )
                quora
                ;;
            company )
                company
                ;;
            city_vehicle )
                city_vehicle
                ;;
            city_vehicle_2 )
                city_vehicle_2
                ;;
        esac
    done
}

Remove() {
    # Here we assume the folder name is the same to the dataset name
    for data in $ARGS
    do  
        echo "Removing ${data}"
        case $data in
            twitter )
                rm -rf  ${DIR}/$data;
                ;;
            quora )
                rm -rf  ${DIR}/$data;
                ;;
            company )
                rm -rf  ${DIR}/$data;
                ;;
            city_vehicle )
                rm -rf  ${DIR}/$data;
                ;;
        esac
    done
}

case "$1" in
    help ) # display Help
        Help
        ;;
    download )
        Download
        ;;
    remove )
        Remove 
        ;;
    + )
        echo "${RED}Usage: check --help${NC}"
        Help
        ;;
    * )
        echo "${RED}Usage: check --help${NC}"
        Help
    ;;
esac
	