#!/bin/bash
#
# This script is the interface to Q2 of this assignment submission.
usage(){
  printf "usage: bash %s <C/R> <train/eval> </path/to/model> </path/to/dataset> [/path/to/val_dataset]\n" "$0" >&2
  printf "       C:    Classification\n       R:    Regression\n" >&2
  printf "       /path/to/model        A Linux style path to the model file.\n">&2
  printf "                             when <train>, model is saved at path,\n" >&2
  printf "                             when eval, model is loaded from path.\n" >&2
  printf "      /path/to/val_dataset   Path to validation dataset.\n" >&2
  printf "                             Will only be given when <train> option is provided.\n" >&2
  exit 1
}
if [ $# -lt 4 ]
then
  usage
fi

task="$1"
train_eval_option="$2"
model_path="$3"
dataset_path="$4"
if [ $# -gt 4 ]
then
  val_dataset_path="$5"
fi
curr_dir="$(pwd)"
#
#
# for classification
if [ "${task}" = "C" ]
then
  cd classification
  if [ "${train_eval_option}" = "train" ]
  then
    if [ $# -lt 5 ]
    then
      usage
    fi
    python3 train.py --model_path "${model_path}" --dataset_path "${dataset_path}" --val_dataset_path "${val_dataset_path}"
  elif [ "${train_eval_option}" = "eval" ]
  then
    python3 evaluate.py --model_path "${model_path}" --dataset_path "${dataset_path}"
  else
    usage
  fi
  cd "${curr_dir}" # back to original DIRECTORY
# for regression
elif [ "${task}" = "R" ]
then
  cd regression
  if [ "${train_eval_option}" = "train" ]
  then
    if [ $# -lt 5 ]
    then
      usage
    fi
    python3 train.py --model_path "${model_path}" --dataset_path "${dataset_path}" --val_dataset_path "${val_dataset_path}"
  elif [ "${train_eval_option}" = "eval" ]
  then
    python3 evaluate.py --model_path "${model_path}" --dataset_path "${dataset_path}"
  else
    usage
  fi
  cd "${curr_dir}" # back to original DIRECTORY
else
  usage
fi
