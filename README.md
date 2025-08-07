# FilterVectorCode

仓库包含一个针对过滤式近似最近邻搜索（Filtered Approximate Nearest Neighbor Search, Filtered ANNS）的增强实现，基于两篇论文中提出的核心思想：**UNG (Unified Navigating Graph)** 和 **ACORN**。


## 1. Filtered Approximate Nearest Neighbor Search

近似最近邻搜索用于从海量高维向量数据中高效地检索出与查询向量最相似的向量 。然而，在许多现实世界的应用中，用户不仅希望找到语义上相似的内容，还希望结果能满足特定的元数据（metadata）约束，例如产品的品牌、论文的发表年份或图片的标签等。

过滤式 ANNS (Filtered ANNS)正是为了解决这一需求而生。它将向量的相似性搜索与结构化数据的属性过滤结合起来，旨在从满足特定过滤条件的向量子集中，找出与查询向量最相似的结果。

## 2. UNG

[Navigating Labels and Vectors: A Unified Approach to Filtered Approximate Nearest Neighbor Search](https://dl.acm.org/doi/10.1145/3698822) 论文提出了 UNG 算法。

### 核心思想

UNG 的核心在于采用“先过滤后搜索” 的思想。关键创新点包括：

* **向量分组 (Vector Partitioning)**：UNG 首先根据向量所关联的标签集 (label set) 对所有数据进行分区。拥有完全相同标签集的向量会被分到同一个组中 。

* **标签导航图 (Label Navigating Graph, LNG)**：UNG 构建一个有向无环图（DAG）称为 LNG，其中的每个节点代表一个唯一的标签集 。如果标签集 `f'` 是 `f` 的“最小超集”（Minimum Superset），那么在 LNG 中就存在一条从 `f` 指向 `f'` 的边。LNG图编码了所有标签集之间的包含关系。

* **统一导航图 (Unified Navigating Graph, UNG)**：最终的图结构是一个统一体,包含：
    1.  **组内邻近图 (Intra-group Proximity Graphs)**：在每个向量分组内部，使用HNSW 或 Vamana 等标准的图算法构建一个 ANNS 邻近图。
    2.  **跨组边 (Cross-group Edges)**：为了连接不同的分组，UNG 依据 LNG 中的连接关系，在不同分组的向量之间添加跨组边。这些边使得搜索可以从一个标签集覆盖的向量群体“导航”到其超集所覆盖的向量群体。

### 工作流程

当一个过滤式 ANNS 查询到来时，UNG 的处理流程如下：
1.  根据查询标签 `fq`，通过 LNG 快速找到所有满足条件的入口标签集 (Entry Label Sets) 。
2.  从这些入口标签集对应的向量组中选取起始向量（entry vectors）。
3.  在统一导航图上执行贪心搜索，既利用组内邻近图进行高效的局部 ANNS 搜索，也利用跨组边在满足过滤条件的向量空间中进行跳转，最终找到最近邻的结果。


## 3. ACORN

[ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data](https://arxiv.org/abs/2403.04871) 论文提出了 ACORN。

### 核心思想

与 UNG 不同，ACORN 直接改造现有的HNSW，使其能够原生支持过滤查询，而无需预先了解所有可能的过滤条件。其关键创新点包括：

* **谓词无关的稠密图构建 (Predicate-Agnostic Dense Graph Construction)**：为了让任意的谓词子图都保持良好的连通性和导航性，ACORN 在构建索引时就有意地创建了一个比标准 HNSW 更稠密的图。通过在构建过程中为每个节点寻找并存储更多的邻居来实现。


## 4. 代码结构和运行方法

### 4.1 项目结构

   ```
   ├── FilterVectorCode
   │   ├── ACORN
   │   ├── DataTools
   │   ├── thirdparty
   │   ├── UNG
   │   ├── README.md
   │   └── run_two.sh
   ├── FilterVectorData
   │   ├── amazing_file
   │   ├── app_reviews
   │   ├── arxiv
   │   ├── MTG
   │   └── ... (其他数据集)
   └── FilterVectorResults
      ├── ACORN
      ├── merge_results
      ├── UNG
      └── UNG+ACORN
   ```

  * **`FilterVectorCode/`**: 项目的源代码。

      * `ACORN/` 和 `UNG/`: 分别是 ACORN 和 UNG 原始算法的核心实现。
      * `DataTools/`: 存放数据处理、格式转换等相关的工具脚本。
      * `thirdparty/`: 项目依赖的第三方库。
      * `run_two.sh`: 用于执行核心实验或运行主程序的 Shell 脚本。

  * **`FilterVectorData/`**: 实验所需的所有数据集。每个子目录都对应一个特定的数据集，例如 `arxiv`、`MTG`、`tiktok_reviews` 和 `words` 等。

  * **`FilterVectorResults/`**: 存储所有实验运行后产生的结果文件。
      * `ACORN/` 和 `UNG/`: 分别存放原始 ACORN 和 UNG 算法的实验结果。
      * `UNG+ACORN/`: 存放改进后的组合算法的实验结果。
      * `merge_results/`: 用于存放汇总或对比分析后的最终结果。


### 4.2 数据集格式

以 `bookimg` 数据集为例，其目录结构如下：

```
bookimg/
├── base_5/
│   └── bookimg_base_labels.txt
├── base_6/
│   └── ...
├── query_5/
│   ├── bookimg_gt_labels_containment.bin
│   ├── bookimg_query_labels.txt
│   ├── bookimg_query_stats.txt （可选）
│   ├── bookimg_query.bin
│   └── bookimg_query.fvecs
├── query_6/
│   └── ...
├── bookimg_base.bin
└── bookimg_base.fvecs
```

  * **基础数据集 (Base Data)**

      * `bookimg_base.fvecs` / `bookimg_base.bin`: 存储了完整的基础向量数据，通常首先以 `.fvecs`（FAISS 支持的格式），然后使用UNG中代码转化为通用二进制 `.bin` 格式。
      * `base_*/bookimg_base_labels.txt`: 这是一个文本文件，每一行对应一个基础向量的标签信息。每行代表一个向量的属性，使用逗号分隔，示例如下：

         ```
         4993,25133,32265,32275
         16673,23789
         4997,25163,32265
         ```

  * **查询数据集 (Query Data)**

      * `query_*/bookimg_query.fvecs` / `bookimg_query.bin`: 存储了用于测试的查询向量。
      * `query_*/bookimg_query_labels.txt`: 文本文件，存储了每个查询向量所关联的过滤标签。
      * `query_*/bookimg_gt_labels_containment.bin`: 这是二进制格式的Ground Truth文件，由UNG中特定代码文件生成。它记录了在特定过滤条件下（如此处的 `containment` 场景），每个查询向量的真实最近邻向量的ID，是计算Recall的依据。
      * `query_*/bookimg_query_stats.txt`: 包含查询集的统计信息，例如每个查询的过滤选择性（selectivity）等，用于分析算法在不同查询难度下的性能。是生成查询任务时输出的日志。

### 4.3 运行方法
#### 4.3.1 UNG运行方法
UNG由 `exp_ung.sh`、`run.sh` 和 `experiments.json` 三个核心文件协同控制。运行UNG目录下`exp_sh.sh`文件即可，这是运行所有实验的入口。

其中，需要根据自己的目录结构修改以下几个地方：

##### a) 修改 `experiments.json` 配置文件
- data_dir: 请将其修改为存放 FilterVectorData的实际路径。
- output_dir: 请将其修改为存放 FilterVectorResults的实际路径。

##### b) 修改 `exp_ung.sh` 启动脚本
- export TMPDIR: 此环境变量定义了存放临时编译文件的目录。脚本会在这里创建 build 文件夹。请确保对该路径有读写权限。
- build_dir: 这是传递给run.sh的参数，定义了每个数据集独立的编译目录。

   ```bash
   # 修改前
   export TMPDIR="/data/fxy/FilterVector/build"
   # ...
   --build_dir "/data/fxy/FilterVector/build/build_$dataset" \

   # 假如您的项目在 /home/user/my_project下，则修改为：
   export TMPDIR="/home/user/my_project/build_temp"
   # ...
   --build_dir "/home/user/my_project/build_temp/build_$dataset" \
   ```

##### c) 脚本工作流程

1. 读取配置: exp_ung.sh 读取 experiments.json 文件，并使用 jq 逐一解析每个实验的配置。

2. 循环调用: 对于 experiments.json 中定义的每一个实验，exp_ung.sh 都会调用 run.sh 脚本，并将该实验的所有参数通过命令行传递给 run.sh。

3. 编译与执行 (由 run.sh 完成):
   - 为当前数据集创建一个独立的 build 目录。
   - 使用 cmake 和 make 编译 C++ 源代码。
   - 根据需要，执行数据格式转换、生成基准真相 (Ground Truth) 文件。
   - 调用 build_UNG_index.cpp入口文件构建索引。
   - 调用 search_UNG_index.cpp入口文件执行搜索，并评估性能。

4. 保存结果: 所有日志和结果文件都会被保存在 experiments.json 中配置的 output_dir 下，并按照参数自动创建层级分明的子目录。

5. 清理工作: 所有实验完成后，exp_ung.sh 自动删除设置的临时编译目录 TMPDIR。


#### 4.3.2 ACORN运行方法

ACORN由 `exp_acorn.sh`、`run_more_efs.sh` 和 `experiments.json` 三个核心文件协同控制。运行ACORN目录下`exp_acorn.sh`文件即可，这是运行所有实验的入口。

##### a) 修改路径
其中，sh文件和json文件不用修改路径。CMakeLists文件简单修改：

1. `FilterVectorCode/ACORN/CMakeLists.txt`中，`nlohmann_json`文件位置修改为`FilterVectorCode/thirdparty/json-3.10.4.tar.gz`的绝对路径。
2. `FilterVectorCode/ACORN/tests/CMakeLists.txt`中，`googletest`地址修改为`FilterVectorCode/thirdparty/googletest-release-1.12.1.tar.gz`的绝对地址。

##### b) 脚本工作流程

1.  解析配置: 执行 `exp_acorn.sh` 后,脚本读取 `experiments.json` 文件，并为其中定义的每一个实验启动一轮测试。
2.  编译代码: `run_more_efs.sh` 脚本首先会为当前任务创建一个临时的 `build` 目录，并使用 `CMake` 和 `make` 编译 ACORN 的 C++ 测试程序 (`test_acorn`)。
3.  执行测试: 编译成功后，脚本会调用 `test_acorn` 可执行文件。该程序会：
      * 根据参数构建 ACORN 索引。
      * 加载查询数据，并在一系列 `efs` 值上进行搜索测试。
      * 重复多次实验以获得稳定的性能数据。
4.  保存结果: 所有的性能指标（如 QPS, Recall@10 等）会以 `.csv` 格式保存在 `FilterVectorResults/ACORN/` 目录下。
      * 输出目录会根据实验参数自动命名，便于区分和查找。
      * 每个实验的详细运行参数会保存在一个 `experiment_config.txt` 文件中。
      * C++ 程序的详细运行日志会重定向到 `output_log.log` 文件中。

**`FilterVectorCode/ACORN/run_acorn_for_ung.sh`文件是idea2的运行脚本，可以不看。**