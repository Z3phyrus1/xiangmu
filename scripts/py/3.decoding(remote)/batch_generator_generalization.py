# -*- coding: utf-8 -*-
import os
from shutil import copyfile

CONDA_ENV_NAME = "neuro_decoding"

subjects_list = [f"sub{i:02d}" for i in range(18, 29)]   # sub18 ... sub28
trials_list = ["1"]                                       # only trial 1

source_script = "generalization_script.py"
dependencies = ["utils.py", "common_settings.py"]
scripts_output_dir = "../bash_generalization_jobs"

slurm = {
    "job_prefix": "Gen",
    "time": "12:00:00",
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 4,
    "mem": "16G",
    "account": "ChenQi_13"
}

def main():
    os.makedirs(scripts_output_dir, exist_ok=True)
    os.makedirs(os.path.join(scripts_output_dir, "logs"), exist_ok=True)

    copyfile(source_script, os.path.join(scripts_output_dir, source_script))
    for dep in dependencies:
        if not os.path.exists(dep):
            raise FileNotFoundError(f"Missing dependency: {dep}")
        copyfile(dep, os.path.join(scripts_output_dir, dep))

    submit_cmds = []
    for subj in subjects_list:
        for trial in trials_list:
            job_name = f"{slurm['job_prefix']}_{subj}_t{trial}"
            sh_name = f"run_{subj}_{trial}.sh"
            sh_path = os.path.join(scripts_output_dir, sh_name)

            content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.out
#SBATCH --error=logs/{job_name}.err
#SBATCH --time={slurm['time']}
#SBATCH --nodes={slurm['nodes']}
#SBATCH --ntasks={slurm['ntasks']}
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --mem={slurm['mem']}
#SBATCH --account={slurm['account']}

source ~/.bashrc
conda activate {CONDA_ENV_NAME}

cd $SLURM_SUBMIT_DIR
python3 {source_script} --subject {subj} --trial {trial}
"""
            with open(sh_path, "w", encoding="utf-8") as f:
                f.write(content)
            submit_cmds.append(f"sbatch {sh_name}")

    with open(os.path.join(scripts_output_dir, "submit_all_jobs.sh"), "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n" + "\n".join(submit_cmds) + "\n")

    print("Generated successfully.")
    print(f"Next:\n  cd {scripts_output_dir}\n  sh submit_all_jobs.sh")

if __name__ == "__main__":
    main()