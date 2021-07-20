#!/bin/bash

function fetch () {
    case $1 in
        model)
            echo "Fetching model";
            if [ ! -d ./tmp ]
            then
                mkdir tmp;
            fi
            cd tmp;
            wget -q --show-progress https://res.mdpi.com/data/ScilitBERT_plus_tokenizer.zip;
            unzip ScilitBERT_plus_tokenizer.zip
            if [ ! -d ../ScilitBERT ]
            then
                mv ScilitBERT ..
            else 
                echo "ERROR: ScilitBERT directory already exists"
            fi
            cd ..
            rm -r ./tmp
            ;;
        dataset)
            echo "Fetching dataset";
            if [ ! -d ./tmp ]
            then
                mkdir tmp;
            fi
            cd tmp;
            wget -q --show-progress https://res.mdpi.com/data/journal-finder.zip;
            unzip journal-finder.zip
            if [ ! -d ../Journal-Finder ]
            then
                mv Journal-Finder ..
            else 
                echo "ERROR: Journal-Finder directory already exists"
            fi
            cd ..;
            rm -r ./tmp;
            ;;
        *)
            echo "wrong argument";
            ;;
        esac
}

function fetch_and_unzip () {
  case $1 in
    both)
        fetch "model"
        fetch "dataset"
        ;;
    model)
        fetch "model"
        ;;
    dataset)
        fetch "dataset"
        ;;
    *)
        echo "Error in value";
        ;;
  esac;
}

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -t|--target)
    # "both", "dataset", "model"
      echo $2
      fetch_and_unzip $2
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      echo "Error in arguments" 
      shift # past argument
      ;;
  esac;
done