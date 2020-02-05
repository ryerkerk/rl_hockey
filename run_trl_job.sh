#!/bin/bash

date
echo "Starting"

error_exit () {
  echo "${BASENAME} - ${1}" >&2
  exit 1
}

if [ ! -d "results" ]; then
  mkdir results
fi

if [ ! -d "trained_models" ]; then
  mkdir trained_models
fi

if [ ! ${initial_model} == "none" ]; then
  aws s3 cp "s3://rl-hockey/trained_models/${initial_model}.pt" - > "./trained_models/${initial_model}.pt" || error_exit "Failed to download initial model from s3 bucket."
fi

# Check command line arguments for script, and then check if a
# matching envinronmental variable is defined.

# If they do, add to argument list for eventual python call
pat="--([^ ]+).+"
arg_list=""
while IFS= read -r line; do
    # Check if line contains a command line argument
    if [[ $line =~ $pat ]]; then
      E=${BASH_REMATCH[1]}
      # Check that a matching environmental variable is declared
      if [[ ! ${!E} == "" ]]; then
        # Make sure argument isn't already include in argument list
        if [[ ! ${arg_list} =~ "--${E}=" ]]; then
          # Add to argument list
          arg_list="${arg_list} --${E}=${!E}"
        fi
      fi
    fi
done < <(python3 main.py --help)

python3 -u main.py ${arg_list} | tee "${save_name}.txt"

aws s3 cp "./results/${save_name}.p" "s3://rl-hockey/results/${save_name}.p" || error_exit "Failed to upload results to s3 bucket."
aws s3 cp "./trained_models/${save_name}.pt" "s3://rl-hockey/trained_models/${save_name}.pt" || error_exit "Failed to upload results to s3 bucket."
aws s3 cp "./${save_name}.txt" "s3://rl-hockey/logs/${save_name}.txt" || error_exit "Failed to upload logs to s3 bucket."

date
echo "Finished"
