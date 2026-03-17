#!/bin/bash
#SBATCH --reservation=fri
#SBATCH --job-name=code_sample
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=sample_out.log
#SBATCH --hint=nomultithread

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load numactl

# Compile
gcc main.c -O3 -fopenmp -lnuma -lm -o main

mkdir -p ../test_images_out
mkdir -p results

RUNS=5
RESULTS_FILE="results/timings.csv"
SUMMARY_FILE="results/summary.txt"

# pobriši stare rezultate
echo "image,run,time_s" > "$RESULTS_FILE"
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

        # Zajemi celoten izpis programa
        program_output=$(srun ./main "$input" "$output" "$seams" 2>&1)

        # Izlušči čas iz vrstice: Time: 0.123456 s
        time_s=$(echo "$program_output" | awk '/^Time:/ {print $2}')

        # Če čas ni bil najden, javi napako in izpiši output
        if [ -z "$time_s" ]; then
            echo "Napaka: časa nisem našel v izpisu."
            echo "$program_output"
            exit 1
        fi

        echo "$input,$i,$time_s" >> "$RESULTS_FILE"
        echo "    Time = $time_s s"
    done

    # Izračun povprečja za trenutno sliko
    avg=$(awk -F, -v img="$input" '
        $1 == img {sum += $3; count++}
        END {
            if (count > 0) printf "%.6f", sum / count;
        }
    ' "$RESULTS_FILE")

    echo "  Average for $input = $avg s"
    echo "$input -> avg = $avg s" >> "$SUMMARY_FILE"
}

run_and_measure "valve.png"                         "valve-out.png"                         128
run_and_measure "../test_images/720x480.png"       "../test_images_out/720x480-out.png"   128
run_and_measure "../test_images/1024x768.png"      "../test_images_out/1024x768-out.png"  128
run_and_measure "../test_images/1920x1200.png"     "../test_images_out/1920x1200-out.png" 128
run_and_measure "../test_images/3840x2160.png"     "../test_images_out/3840x2160-out.png" 128
run_and_measure "../test_images/7680x4320.png"     "../test_images_out/7680x4320-out.png" 128

echo ""
echo "Rezultati shranjeni v:"
echo "  $RESULTS_FILE"
echo "  $SUMMARY_FILE"