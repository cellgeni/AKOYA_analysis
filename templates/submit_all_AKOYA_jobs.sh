mkdir -p logs

for conf in conf_files/conf_*.yaml; do
    base=$(basename "$conf" .yaml)
    base=${base#conf_}
    job_name="AKOYA_${base}"

    bsub \
        -G team283 \
        -q hugemem \
        -n 1 \
        -M 400000 \
        -R 'select[mem>400000] rusage[mem=400000]' \
        -J "$job_name" \
        -o "logs/output_${job_name}_%J.log" \
        -e "logs/error_${job_name}_%J.log" \
        "module load ISG/conda && \
         conda activate ./sp_env && \
         export PYTHONNOUSERSITE=1 && \
         python bin/run_AKOYA.py '$conf'"
done