threads_per_block = 256

for blocks in [1,2,4,8,16]:
    for n in [2**21, 2**22, 2**23, 2**24, 2**25]:
        for algo in [0,1]:
            # Write the SLURM script
            with open(f"CUDA_scripts/{n}_{threads_per_block}_{blocks}_{algo}.sh", "w") as script_file:
                script_file.write(f"""#!/bin/bash
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name={n}_{threads_per_block}_{blocks}_{algo}
#SBATCH --output={n}_{threads_per_block}_{blocks}_{algo}.out
#SBATCH --error={n}_{threads_per_block}_{blocks}_{algo}.err
srun marzola.out {n} {blocks} {threads_per_block} {algo}
            """)
        
    
#####################################################################################################
        
                
# n=2**20
# for blocks in [1,2,4,8,16]:
#     for threads_per_block in [32,64,128,256,512,1024]:
#         for algo in [0,1]:
#             # Write the SLURM script
#             with open(f"scripts/{n}_{threads_per_block}_{blocks}_{algo}.sh", "w") as script_file:
#                 script_file.write(f"""#!/bin/bash
# #SBATCH --partition=edu5
# #SBATCH --nodes=1
# #SBATCH --tasks=1
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=1
# #SBATCH --time=00:05:00
# #SBATCH --job-name={n}_{threads_per_block}_{blocks}_{algo}
# #SBATCH --output={n}_{threads_per_block}_{blocks}_{algo}.out
# #SBATCH --error={n}_{threads_per_block}_{blocks}_{algo}.err
# srun marzola.out {n} {blocks} {threads_per_block} {algo}
#             """)