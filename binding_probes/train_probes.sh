export PROJECT_DIR=/users/mlepori/data/mlepori/projects/gestalt-interp

model=("scratch_32" "scratch_16" "facebook/dino-vitb16" "facebook/dinov2-base" "facebook/vit-mae-base" "google/vit-base-patch16-224-in21k" "google/vit-base-patch32-224-in21k" "openai/clip-vit-base-patch16" "openai/clip-vit-base-patch32") 
for i in "${!model[@]}"; do
    export MODEL=${model[i]}


    echo "Running $MODEL"

    sbatch -o out/1_${i}.out -e err/1_${i}.err $PROJECT_DIR/binding_probes/train_probes.script
done

