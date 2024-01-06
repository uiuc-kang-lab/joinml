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
   echo "twitter                      twitter dataset (about 8.3MB)"
   echo "stackoverflow                stackoverflow dataset (about 51.7MB)"
   echo "quora                        quora dataset (about 4.9MB)"
   echo "company                      company dataset (about 129.7MB)"
   echo "city_vehicle                 AICity Vehicle multi-camera multi-target tracking dataset (about 189.3MB)"
   echo "city_human                   AICity Human multi-camera multi-target tracking dataset (about 1.42GB)"
   echo "flickr30k                    multi-modal flickr 30K dataset (4.15GB)"
}

twitter()
{
    if [ ! -d "${DIR}/twitter/" ];
    then
        echo "Downloading twitter dataset (about 8.3MB)..."
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
        echo "Downloading quora dataset (about 4.9MB)..."
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

stackoverflow()
{
    if [ ! -d "${DIR}/stackoverflow/" ];
    then
        echo "Downloading stackoverflow dataset (about 51.7MB)..."
        gdown 1m6DUQdXrAjT-ku2ppB5gV4TEGf6t_K5N

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/stackoverflow.tar -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/stackoverflow.tar

        echo -e "${GREEN}stackoverflow dataset downloaded!${NC}"
    else
        echo -e "${RED}stackoverflow dataset already exists under ${DIR}/stackoverflow/!"
fi
}

company()
{
    if [ ! -d "${DIR}/company/" ];
    then
        echo "Downloading company dataset (about 129.7MB)..."
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
        echo "Downloading AICity vehicle multi-camera multi-target tracking dataset (about 189.3MB)..."
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

city_human()
{
    if [ ! -d "${DIR}/city_human/" ];
    then
        echo "Downloading AICity vehicle multi-camera multi-target tracking dataset (about 1.42GB)..."
        gdown 1P5-CrcXxxzqVhiO-9Rw80Q-ypMFjXOnj

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/city_human.tar -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/city_human.tar

        echo -e "${GREEN}AICity human multi-camera multi-target tracking dataset downloaded!${NC}"
    else
        echo -e "${RED}AICity human multi-camera multi-target tracking dataset already exists under ${DIR}/city_human/!"
fi
}

flickr30k()
{
    if [ ! -d "${DIR}/flickr30k/" ];
    then
        echo "Downloading flickr30k dataset (about 4.15GB)..."
        gdown 1H3gjm-i0-9fr0YpXcYIJsQjY2dfWEY83

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/flickr30k.tar -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/flickr30k.tar

        echo -e "${GREEN}flickr30k dataset downloaded!${NC}"
    else
        echo -e "${RED}flickr30k dataset already exists under ${DIR}/flickr30k/!"
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
            stackoverflow )
                stackoverflow
                ;;
            company )
                company
                ;;
            city_vehicle )
                city_vehicle
                ;;
            city_human )
                city_human
                ;;
            flickr30k )
                flickr30k
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
            stackoverflow )
                rm -rf  ${DIR}/$data;
                ;;
            city_vehicle )
                rm -rf  ${DIR}/$data;
                ;;
            city_human )
                rm -rf  ${DIR}/$data;
                ;;
            flickr30k )
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
	