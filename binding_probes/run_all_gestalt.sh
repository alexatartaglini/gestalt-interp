export PROJECT_DIR=/users/mlepori/data/mlepori/projects/gestalt-interp/
export CONFIG_DIR=configs/clip_16

for file in ${CONFIG_DIR}/*;
do
    echo "$file"
    export JOBNAME=$(basename "$file" .yaml)
    export CONFIG=/users/mlepori/data/mlepori/projects/gestalt-interp/binding_probes/${CONFIG_DIR}/${JOBNAME}.yaml
    sbatch -J $JOBNAME -o out/${JOBNAME}.out -e err/${JOBNAME}.err $PROJECT_DIR/binding_probes/run_all_gestalt.script
done

