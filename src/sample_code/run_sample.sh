#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=code_sample
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=sample_out.log
#SBATCH --hint=nomultithread

# Ustavi izvajanje skripte, če kateri koli ukaz (npr. gcc) spodleti!
set -e 

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load numactl

# Compile - TUKAJ JE POPRAVEK! Dodan je utils.c
# (Če imaš utils.c prav tako v mapi algorithm_implementations, 
# potem uporabi pot: ./algorithm_implementations/utils.c)
gcc -O3 -fopenmp ./algorithm_implementations/parallel_seam.c ./algorithm_implementations/utils.c -o parallel_seam -lm -lnuma
# Tukaj ugasnemo "set -e", da nam skripta ne crkne, če slučajno srun vrne opozorilo
set +e

mkdir -p ../test_images_out
mkdir -p results

RUNS=5
THREADS=$OMP_NUM_THREADS
RESULTS_FILE="results/timings-parallel_seam.csv"
SUMMARY_FILE="results/summary-parallel_seam.txt"

echo "image,threads,run,time_s" > "$RESULTS_FILE"
echo "POVZETEK MERITEV" > "$SUMMARY_FILE"
echo "================" >> "$SUMMARY_FILE"

run_and_measure () {
    local input="$1"
    local output="$2"
    local seams="$3"

    echo ""
    echo "Testing: $input"

    for ((i=1; i<=RUNS; i++)); do
        echo "  Run $i/$RUNS"

        program_output=$(srun ./parallel_seam "$input" "$output" "$seams" 2>&1)
        time_s=$(echo "$program_output" | awk '/^Time:/ {print $2}')

        if [ -z "$time_s" ]; then
            echo "Napaka: časa nisem našel v izpisu."
            echo "$program_output"
            exit 1
        fi

        echo "$input,$THREADS,$i,$time_s" >> "$RESULTS_FILE"
        echo "    Time = $time_s s"
    done

    avg=$(awk -F, -v img="$input" -v th="$THREADS" '
        $1 == img && $2 == th {sum += $4; count++}
        END {
            if (count > 0) printf "%.6f", sum / count;
        }
    ' "$RESULTS_FILE")

    echo "  Average for $input = $avg s"
    echo "$input | threads=$THREADS | avg=$avg s" >> "$SUMMARY_FILE"
}

run_and_measure "valve.png"                     "valve-out.png"                         128
run_and_measure "../test_images/720x480.png"   "../test_images_out/720x480-out.png"   128
run_and_measure "../test_images/1024x768.png"  "../test_images_out/1024x768-out.png"  128
run_and_measure "../test_images/1920x1200.png" "../test_images_out/1920x1200-out.png" 128
run_and_measure "../test_images/3840x2160.png" "../test_images_out/3840x2160-out.png" 128
run_and_measure "../test_images/7680x4320.png" "../test_images_out/7680x4320-out.png" 128

echo ""
echo "Rezultati shranjeni v:"
echo "  $RESULTS_FILE"
echo "  $SUMMARY_FILE"