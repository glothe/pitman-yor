#!/usr/bin/env bash

DATA_DIR=data
DATA_FILE=thyroid_train.dat
DATA_URL="https://www.di.ens.fr/~cappe/fr/Enseignement/data/thyroid_train.dat"

mkdir -p $DATA_DIR
wget $DATA_URL -O $DATA_DIR/$DATA_FILE

