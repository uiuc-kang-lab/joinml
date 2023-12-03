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
   echo "twitter                      twitter dataset (about 10.3MB)"
   echo "quora                        quora dataset (about 4.9MB)"
   echo "stackoverflow                stackoverflow dataset (about 51.6MB)"
   echo "company                      company dataset (about 129.5MB)"
   echo "city_vehicle                 AICity Vehicle multi-camera multi-target tracking dataset (about 469MB)"
   echo "city_vehicle_2               AICity Vehicle multi-camera multi-target tracking dataset two table version (about 336MB)"
   echo "city_human                   AICity Human multi-camera multi-target tracking dataset (about 583.4MB)"
   echo "======= Available models ========="
   echo "twitter_minilm                 finetuned bi-encoder for twitter dataset with minilm l6 v2 (about 79.7MB)"
   echo "quora_minilm                   finetuned bi-encoder for quora dataset with minilm l6 v2 (about 79.7MB)"
   echo "company_minilm                 finetuned bi-encoder for company dataset with minilm l6 v2 (about 79.7MB)"
   echo "stackoverflow_proxy            finetuned bi-encoder for stackoverflow dataset (about 87.6MB)"
}

twitter()
{
    if [ ! -d "${DIR}/twitter/" ];
    then
        echo "Downloading twitter dataset(about 10.3MB)..."
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
        echo "Downloading quora dataset(about 4.9MB)..."
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
        echo "Downloading stackoverflow dataset(about 51.6MB)..."
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

city_human()
{
    if [ ! -d "${DIR}/city_human/" ];
    then
        echo "Downloading AICity vehicle multi-camera multi-target tracking dataset(about 584.3MB)..."
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

quora_minilm()
{
    if [ ! -d "${DIR}/quora-MiniLM-L6-v2/" ];
    then
        echo "Downloading finetuned model for quora (about 79.9MB)..."
        gdown 1ch9xDDnLWkx0TcIsK1ckRcWfK3h-10Oj

        echo "Dataset downloaded, now decompressing..."
        unzip ${DIR}/quora-MiniLM-L6-v2.zip

        echo "Removing compressed file..."
        rm -f ${DIR}/quora-MiniLM-L6-v2.zip

        echo -e "${GREEN}Finetuned model for quora downloaded!${NC}"
    else
        echo -e "${RED}Finetuned model for quora already exists under ${DIR}/quora-MiniLM-L6-v2/!"
fi
}

twitter_minilm()
{
    if [ ! -d "${DIR}/twitterall-MiniLM-L6-v2/" ];
    then
        echo "Downloading finetuned model for twitter (about 79.9MB)..."
        gdown 1Fxw7xPMKhOtlMvjXnIOoJy1jvMH8B6Hi

        echo "Dataset downloaded, now decompressing..."
        unzip ${DIR}/twitterall-MiniLM-L6-v2.zip

        echo "Removing compressed file..."
        rm -f ${DIR}/twitterall-MiniLM-L6-v2.zip

        echo -e "${GREEN}Finetuned model for twitter downloaded!${NC}"
    else
        echo -e "${RED}Finetuned model for twitter already exists under ${DIR}/twitterall-MiniLM-L6-v2/!"
fi
}

stackoverflow_proxy()
{
    if [ ! -d "${DIR}/stackoverflow_proxy/" ];
    then
        echo "Downloading finetuned model for stackoverflow (about 79.9MB)..."
        gdown 1JlP7mnZaM7gb96Q8ZNy8ivflAm5zS93G

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/stackoverflow_proxy.tar -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/stackoverflow_proxy.tar

        echo -e "${GREEN}Finetuned model for stackoverflow downloaded!${NC}"
    else
        echo -e "${RED}Finetuned model for stackoverflow already exists under ${DIR}/stackoverflow_proxy/!"
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
            city_vehicle_2 )
                city_vehicle_2
                ;;
            city_human )
                city_human
                ;;
            quora_minilm )
                quora_minilm
                ;;
            twitter_minilm )
                twitter_minilm
                ;;
            stackoverflow_proxy )
                stackoverflow_proxy
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
	