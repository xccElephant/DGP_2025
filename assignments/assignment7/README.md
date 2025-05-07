# 平均值坐标（Mean Value Coordinates）

> Michael S. Floater,
> 
> Mean value coordinates,
> 
> Computer Aided Geometric Design,
> 
> Volume 20, Issue 1,
> 
> 2003,
> 
> Pages 19-27,
> 
> ISSN 0167-8396,
> 
> https://doi.org/10.1016/S0167-8396(03)00002-5.
> 
> (https://www.sciencedirect.com/science/article/pii/S0167839603000025)
> 
> Abstract: We derive a generalization of barycentric coordinates which allows a vertex in a planar triangulation to be expressed as a convex combination of its neighbouring vertices. The coordinates are motivated by the Mean Value Theorem for harmonic functions and can be used to simplify and improve methods for parameterization and morphing.
> 
> Keywords: Barycentric coordinates; Harmonic function; Mean value theorem; Parameterization; Morphing


## 实验步骤

### 1. Git 拉取仓库更新

对于一般的Git仓库，只需执行`git pull`命令，即可拉取仓库最新的更新。

然而，本实验框架仓库是包含子模块(`git submodule`)的目录，执行`git pull`后，只会更新直接包含在仓库中的文件，而不会更新仓库的子模块。

如果要确保仓库中所有内容都最新，需要在拉取后，再更新子模块。命令如下：

```shell
git pull    # 拉取仓库更新
git submodule update --init --recursive    # 更新子模块
```

这里提供一种更方便的方式。以下命令可以为给git添加一个新命令`git pullall`，执行该命令时，会自动拉取仓库更新并更新子模块。

```shell
git config --global alias.pullall '!f(){ git pull "$@" && git submodule update --init --recursive; }; f'
```

执行完上述命令后，只需执行

```shell
git pullall
```

即可拉取仓库更新并更新子模块。

### 2. 编译运行

拉取完成后，使用CMake配置项目的同学，最好重新执行一次CMake，以确保新添加的文件被正确添加到项目中。

使用VS文件夹模式打开项目的同学，可以直接重新用VS打开，VS会自动重新进行CMake配置。

打开VS之后，按下`Ctrl+Shift+B`完整编译项目，然后按下`F5`运行项目即可。

**万一出现无法正确运行的问题，尝试“生成”-“重新生成解决方案”/“全部重新生成”。**

### 3. 实现平均值坐标的计算

你需要根据论文的内容，以及代码中的注释（其中包含ARAP Deformation算法介绍），完成`source/Editor/geometry_nodes/node_mvc.cpp`中包含`TODO`的部分（补全第69行的lambda表达式，删去原来的占位代码）。

![image](../../images/assignment_7_2.png)

平均值坐标的表达式：

$$
\lambda_i = \frac{w_i}{\sum_{j=1}^{k} w_j}, \; w_i = \frac{\tan{(\alpha_{i-1}/2)}+\tan{(\alpha_{i}/2)}}{\left\|v_i-v_0\right\|}
$$

**注意边界上的点的处理**。

### 4. 测试平均值坐标的计算

将`assignment/assignment7`目录下的`stage.usdc`文件放到`Assets`目录下，然后运行程序，可以看到`Stage Viewer`窗口中出现了`mesh_0`项，右击`mesh_0`项，选择`Edit`打开节点编辑器后，所有节点自动由下向上计算，将网格显示在`Polyscope Renderer`窗口中。然后请手动在`Polyscope Structure Info`窗口中，打开`/mesh_0`的`function`项，可视化计算出的结果。

调整`random_2d_polygon`节点参数，可生成随机的多边形；`mvc`节点输出一个由二维坐标为输入，以平均值坐标为输出的节点；`function_decompose`节点将平均值坐标分解为对$v_{index}$顶点的权重，即$\omega_{index}$，`visualize_2d_function`将权重可视化，其中` fineness`参数决定采样的精度。

![image](../../images/assignment_7_1.png)

## 实验提交

将`node_arap_deformation.cpp`打包为 `zip` 文件，并将其命名为 `学号_姓名_hw6.zip`，通过邮件发送至 `hwc20040629@mail.ustc.edu.cn`，在邮件主题中注明课程名称、作业序号和学号、姓名。
