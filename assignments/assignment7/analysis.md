# 平均值坐标（Mean Value Coordinates）

## 平均值坐标定义

<img src="../../images/assignment_7_2.png" alt="image" style="zoom:50%;" />
$$
\lambda_i = \frac{w_i}{\sum_{j=1}^{k} w_j}, \; w_i = \frac{\tan{(\alpha_{i-1}/2)}+\tan{(\alpha_{i}/2)}}{\left\|v_i-v_0\right\|}
$$

- 若$v_0$在多边形内部，该式没有歧义

- 若$v_0$在多边形边$(v_{i-1}, v_{i})$上，则$\alpha_{i-1} = \pi$，$\tan{(\alpha_{i-1}/2)} = +\infty$，这会使得$w_{i-1}$和$w_{i}$都趋向于无穷大，而其他顶点的权重$w$仍为常数级别。
    - 最终的坐标值为：$\lambda_{i-1} = \frac{\left\|v_{i}-v_0\right\|}{\left\|v_{i}-v_{i-1}\right\|}$，$\lambda_{i} = \frac{\left\|v_{i-1}-v_0\right\|}{\left\|v_{i}-v_{i-1}\right\|}$，其他顶点的$\lambda$值为$0$
    - 它的几何意义是：对线段上的点，只需考虑距离线段两个端点的距离，即可用这两个端点的坐标表示

- 若$v_0$正好是多边形的顶点$v_i$，则显然$\lambda_i = 1$，其他顶点的$\lambda$值为$0$

## 作业问题描述

补全以下函数（Lambda表达式），作为节点的输出，满足以下条件：
1. 函数捕获一个数组`polygon_vertices`，其中按顺序存储多边形的顶点坐标
2. 函数的输入是一个点的坐标`p_x`和`p_y`
3. 函数的输出是一个数组，其中按顺序存储对多边形各个顶点的平均值坐标


```cpp
    auto mvc_function = [captured_vertices = polygon_vertices](
                            float p_x, float p_y) -> std::vector<float> {
        // 创建辅助变量
        int num_vertices = captured_vertices.size();
        std::vector<float> distances(num_vertices);
        std::vector<float> tan_half(num_vertices);
        std::vector<float> weights(num_vertices);

        // 计算点p到每个顶点的距离
        for (int i = 0; i < num_vertices; ++i) {
            float x_i = captured_vertices[i][0];
            float y_i = captured_vertices[i][1];
            distances[i] = std::sqrt(
                (x_i - p_x) * (x_i - p_x) + (y_i - p_y) * (y_i - p_y));

            if (distances[i] < 1e-5) {
                // 如果点p到某个顶点的距离非常小，认为p就在该顶点上，直接返回平均值坐标
                std::vector<float> weights(num_vertices, 0.0f);
                weights[i] = 1.0f;
                return weights;
            }
        }

        // 计算每个半角的tangent值
        for (int i = 0; i < num_vertices; ++i) {
            float x_i = captured_vertices[i][0];
            float y_i = captured_vertices[i][1];
            int next_i = (i + 1) % num_vertices;
            float x_next = captured_vertices[next_i][0];
            float y_next = captured_vertices[next_i][1];
            float opposite_length = std::sqrt(
                (x_next - x_i) * (x_next - x_i) +
                (y_next - y_i) * (y_next - y_i));
            float cos_i = distances[i] * distances[i] +
                          distances[next_i] * distances[next_i] -
                          opposite_length * opposite_length;
            cos_i /= (2 * distances[i] * distances[next_i]);
            // 处理数值不稳定的情况
            if (std::abs(cos_i - 1.0f) < 1e-5) {
                cos_i = 1.0f - 1e-5;
            }
            // 处理点p在边上，alpha为pi的情况
            else if (std::abs(cos_i + 1.0f) < 1e-5) {
                cos_i = -1.0f + 1e-5;
                // 也可以直接在这里计算该情况的结果，直接返回
            }
            // 半倍角公式
            tan_half[i] = std::sqrt((1 - cos_i) / (1 + cos_i));
        }

        // 计算平均值坐标
        for (int i = 0; i < num_vertices; ++i) {
            int prev_i = (i - 1 + num_vertices) % num_vertices;
            weights[i] = (tan_half[prev_i] + tan_half[i]) / distances[i];
        }

        // 归一化
        float sum_weights = 0.0f;
        for (int i = 0; i < num_vertices; ++i) {
            sum_weights += weights[i];
        }
        for (int i = 0; i < num_vertices; ++i) {
            weights[i] /= sum_weights;
        }
        return weights;
    };
```