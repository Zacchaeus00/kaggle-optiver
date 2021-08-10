#!/bin/bash
#SBATCH --job-name=lofo-oof-recursive           # 任务名
#SBATCH --nodes=1                   # 这里不用动 多节点脚本请查官方文档
#SBATCH --ntasks=1                  # 这里不用动 多任务脚本请查官方文档
#SBATCH --cpus-per-task=16           # 要几块CPU (一般4块就够用了)
#SBATCH --mem=32GB                  # 最大内存
#SBATCH --time=96:00:00             # 运行时间上限
#SBATCH --mail-type=ALL             # ALL / END
#SBATCH --mail-user=yw3642@nyu.edu  # 结束之后给哪里发邮件
#SBATCH --output=lofo-oop-recursive.out         # 正常输出写入的文件
#SBATCH --error=lofo-oop-recursive.err          # 报错信息写入的文件
# #SBATCH --gres=gpu:1                # 需要几块GPU (同时最多8块)
#SBATCH --constraint=cpu # use this if you want to only use cpu
#SBATCH -p parallel                   # 有GPU的partition

lscpu
module purge                        # 清除所有已加载的模块
module load anaconda3               # 加载anaconda (load virtual env for training)

# cd /gpfsnyu/scratch/[Net ID]/project/folder     # 切到程序目录

echo "start pruning"               # 输出起始信息
source deactivate
source activate kaggle          # 调用 virtual env
python -u lofo-oop-recursive.py                     # 用python跑代码
echo "FINISH"                       # 输出起始信息
