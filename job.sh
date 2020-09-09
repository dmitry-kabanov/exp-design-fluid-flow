#!/usr/local_rwth/bin/zsh
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-01:00:00
#SBATCH --job-name=dedalusKH
#SBATCH --output=%J.log

# Load Intel Python Distribution.
module load pythoni/3.7
 
# >>> conda initialize >>>
_root="$(dirname $(dirname $(dirname $(which python))))"
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('${_root}/intelpython3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${_root}/intelpython3/etc/profile.d/conda.sh" ]; then
        . "${_root}/intelpython3/etc/profile.d/conda.sh"
    else
        export PATH="${_root}/intelpython3/bin:$PATH"
    fi
fi
unset __conda_setup
unset _root
# <<< conda initialize <<

# Environment for the current project.
conda activate dedalus

# mpiexec -n $SLURM_NTASKS python ./momentum_mass_balances_0g.py
mpiexec -n 48 python ./momentum_mass_balances_0g.py
