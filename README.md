# ai-image-classifier
git checkout -b feature

pip install scipy matplotlib pandas numpy scikit‑learn


# Step1 安装 DVC，初始化 DVC，Git 提交初始化
# 1.安装dvc
pip install dvc

# 2.初始化git（如果项目还没有git仓库）
git init

# 3.初始化DVC
dvc init

git add .dvc/ .dvcignore .gitignore
git commit -m "Initialize DVC"

# Step2 使用 DVC 添加数据文件做版本管理
# 将原始数据集交给DVC管理
dvc add data/raw/dataset.csv

# 提交 DVC 元文件到 Git：
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "dvc: track raw dataset.csv"


# Step3 配置 DVC 远程存储，推送数据
# 3.1 创建一个本地 DVC 远程目录
# windows cmd
mkdir D:\dvc_remote_storage

# 3.2 添加 DVC 远程，命名为 myremote
dvc remote add -d myremote D:\dvc_remote_storage

git add .dvc/config
git commit -m "dvc: configure default local remote storage"

# 3.3 推送 DVC 缓存数据到远程存储
dvc push

# Step4 DVC 数据版本切换、恢复操作

# 4.1 修改数据集，生成新版本
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc
git commit -m "dvc: update dataset version v2"
dvc push

# 4.2 切换回旧版本数据
# git切换到旧的commit（代码+DVC元文件回退）
git checkout <旧commit_id>
# DVC根据.dvc元文件恢复对应版本真实数据
dvc pull

# 4.3 恢复到最新版本
git checkout main
dvc pull

# Step5 Git + DVC 联合版本管理（实训重点）

# 1. 修改python代码(data_pipeline.py)，同时更新数据集
# 2. DVC处理数据变更
dvc add data/raw/dataset.csv
dvc push

# 3.git提交代码 + dvc元文件
git add data/raw/dataset.csv.dvc data_pipeline.py .dvc/config
git commit -m "code update + dataset v3
