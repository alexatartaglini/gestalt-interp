export PROJECT_DIR=/users/mlepori/data/mlepori/projects/gestalt-interp

model=("scratch_32" "scratch_16") 
for i in "${!model[@]}"; do
    export MODEL=${model[i]}


    echo "Running $MODEL"

    sbatch -o out/1_${i}.out -e err/1_${i}.err $PROJECT_DIR/binding_probes/train_probes.script
done

