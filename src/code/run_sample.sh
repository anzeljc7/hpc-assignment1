#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=code_sample
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=sample_out.log
#SBATCH --hint=nomultithread

PROGRAM_NAME="basic"

set -e 

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load numactl

gcc -O3 -fopenmp "./algorithm_implementations/${PROGRAM_NAME}.c" ./algorithm_implementations/utils.c -o "${PROGRAM_NAME}" -lm -lnuma

set +e

mkdir -p ../test_images_out
mkdir -p results

RUNS=5
THREADS=$OMP_NUM_THREADS

RESULTS_FILE="results/timings-${PROGRAM_NAME}.csv"
SUMMARY_FILE="results/summary-${PROGRAM_NAME}.txt"

echo "slika;niti;zagon;cas_s" > "$RESULTS_FILE"
echo "POVZETEK MERITEV" > "$SUMMARY_FILE"
echo "================" >> "$SUMMARY_FILE"

run_and_measure () {
    local input="$1"
    local output="$2"
    local seams="$3"

    echo ""
    echo "Testiram: $input"

    for ((i=1; i<=RUNS; i++)); do
        echo "  Zagon $i/$RUNS"

        program_output=$(srun "./${PROGRAM_NAME}" "$input" "$output" "$seams" 2>&1)
        time_s=$(echo "$program_output" | awk '/^Time:/ {print $2}')

        if [ -z "$time_s" ]; then
            echo "Napaka: časa nisem našel v izpisu."
            echo "$program_output"
            exit 1
        fi

        echo "$input;$THREADS;$i;$time_s" >> "$RESULTS_FILE"
        echo "    Čas = $time_s s"
    done

    avg=$(awk -F';' -v img="$input" -v th="$THREADS" '
        $1 == img && $2 == th {sum += $4; count++}
        END {
            if (count > 0) printf "%.6f", sum / count;
        }
    ' "$RESULTS_FILE")

    echo "  Povprečje za $input = $avg s"
    echo "$input | niti=$THREADS | povprecje=$avg s" >> "$SUMMARY_FILE"
}

run_and_measure "valve.png"                    "valve-out.png"                        128
run_and_measure "../test_images/720x480.png"   "../test_images_out/720x480-out.png"   128
run_and_measure "../test_images/1024x768.png"  "../test_images_out/1024x768-out.png"  128
run_and_measure "../test_images/1920x1200.png" "../test_images_out/1920x1200-out.png" 128
run_and_measure "../test_images/3840x2160.png" "../test_images_out/3840x2160-out.png" 128
run_and_measure "../test_images/7680x4320.png" "../test_images_out/7680x4320-out.png" 128

echo ""
echo "Rezultati shranjeni v:"
echo "  $RESULTS_FILE"
echo "  $SUMMARY_FILE"