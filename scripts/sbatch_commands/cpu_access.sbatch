#!/bin/bash
#SBATCH -t 0-01:00:00                    # time limit"days-hours:minutes:seconds"
#SBATCH --mem=30G                        # reserve 16GB of memory
#SBATCH -J cpu                          # the job name
#SBATCH --mail-type=END,FAIL,TIME_LIMIT  # send notification emails
#SBATCH --ntasks=1                       # total number of tasks across all nodes
#SBATCH --cpus-per-task=3                # use that many cpu cores
#SBATCH -N 1                             # request slots on 1 node
#SBATCH --error=job_%j_err.txt           # Inside this file you will find the URL
#SBATCH --output=job_%j_out.txt          # capture output
##SBATCH -C "XEON_E5_2640v3"

sleep 1h # keep the job alive for a given period of time
