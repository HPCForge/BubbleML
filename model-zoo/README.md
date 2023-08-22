# Optical Flow Models
## Model Checkpoints
The pretrained models can be downloaded by following the instructions in the official repositories [RAFT](https//github.com/princeton-vl/RAFT) and [GMFlow](https//github.com/haofeixu/gmflow)
The links for the finetuned models are given below. Each link contains the C+B, C+T+B and C+T+S+B model checkpoints respectively.

| Models | Size |
|----|----|
| [RAFT Flow Boiling Gravity Models](https//bubbleml-model-checkpoints.s3.us-east-2.amazonaws.com/raft-flow-boiling-gravity.tar.gz) | 56.0 MB |
| [RAFT Flow Boiling InletVelocity Models](https//bubbleml-model-checkpoints.s3.us-east-2.amazonaws.com/raft-flow-boiling-velscale.tar.gz) | 56.0 MB |
| [RAFT Pool Boiling SubCooled Models](https//bubbleml-model-checkpoints.s3.us-east-2.amazonaws.com/raft-pool-boiling-subcooled.tar.gz) | 56.0 MB |
| [RAFT Pool Boiling Saturated Models](https//bubbleml-model-checkpoints.s3.us-east-2.amazonaws.com/raft-pool-boiling-saturated.tar.gz) | 56.0 MB |
| [RAFT Pool Boiling Saturated Models](https//bubbleml-model-checkpoints.s3.us-east-2.amazonaws.com/raft-pool-boiling-saturated.tar.gz) | 56.0 MB |
| [RAFT Pool Boiling Gravity Models](https//bubbleml-model-checkpoints.s3.us-east-2.amazonaws.com/raft-pool-boiling-gravity.tar.gz) | 56.0 MB |
| [GMFlow Flow Boiling Gravity Models](https//bubbleml-model-checkpoints.s3.us-east-2.amazonaws.com/gmflow-flow-boiling-gravity.tar.gz) | 147.3 MB |
| [GMFlow Flow Boiling InletVelocity Models](https//bubbleml-model-checkpoints.s3.us-east-2.amazonaws.com/gmflow-flow-boiling-velscale.tar.gz) | 146.7 MB |
| [GMFlow Pool Boiling SubCooled Models](https//bubbleml-model-checkpoints.s3.us-east-2.amazonaws.com/gmflow-pool-boiling-subcooled.tar.gz) | 148.0 MB |
| [GMFlow Pool Boiling Saturated Models](https//bubbleml-model-checkpoints.s3.us-east-2.amazonaws.com/gmflow-pool-boiling-saturated.tar.gz) | 148.9 MB |
| [GMFlow Pool Boiling Gravity Models](https//bubbleml-model-checkpoints.s3.us-east-2.amazonaws.com/gmflow-pool-boiling-gravity.tar.gz) | 147.9 MB |

## Model Benchmarks

### Pool Boiling Saturated

|       Model      | Method | Chairs (Val) | Things (Val) | Sintel (Train) - Clean | Sintel (Train) - Final | KITTI (Train) - F1-EPE | KITTI (Train) - F1-all | Boiling (Test) |
|----------------|------|------------|------------|----------------------|----------------------|----------------------|----------------------|--------------|
|        C         |  RAFT  |     0.82     |     9.03     |          2.19          |          4.49          |          9.83          |         37.57          |      4.20      |
|        C         | GMFlow |     0.92     |    10.23     |          3.22          |          4.43          |         17.82          |         56.14          |      4.73      |
|       C+B        |  RAFT  |     0.91     |    11.22     |          2.55          |          5.16          |         13.7           |         44.44          |     **2.33**   |
|       C+B        | GMFlow |     1.31     |    11.99     |          3.78          |          5.12          |         21.91          |         63.04          |     **2.36**   |
|       C+T        |  RAFT  |     1.15     |     4.39     |          1.40          |          2.71          |          5.02          |         17.46          |      4.72      |
|       C+T        | GMFlow |     1.26     |     3.48     |          1.50          |          2.96          |         11.60          |         35.62          |      7.98      |
|      C+T+B       |  RAFT  |     1.28     |     7.69     |          1.69          |          2.95          |          9.96          |         23.61          |      2.38      |
|      C+T+B       | GMFlow |     1.39     |     3.88     |          1.61          |          2.91          |         14.49          |         43.09          |      2.51      |
|      C+T+S       |  RAFT  |     1.21     |     4.69     |          0.77          |          1.22          |          1.54          |          5.64          |      8.39      |
|      C+T+S       | GMFlow |     1.53     |     4.09     |          0.95          |          1.28          |          3.04          |         13.61          |     14.65      |
|     C+T+S+B      |  RAFT  |     1.37     |     6.59     |          0.89          |          1.60          |          1.83          |          6.44          |      2.34      |
|     C+T+S+B      | GMFlow |     1.65     |     4.49     |          1.07          |          1.45          |          4.06          |         18.99          |      2.56      |

### Pool Boiling SubCooled

|    Model    | Method | Chairs(Val) | Things(Val) | Sintel(Train) - Clean | Sintel(Train) - Final | KITTI(Train) - F1-EPE | KITTI(Train) - F1-all | Boiling(Test) |
|-----------|------|-----------|-----------|---------------------|---------------------|---------------------|---------------------|-------------|
|      C      |  RAFT  |    0.82     |    9.03     |         2.19          |         4.49          |         9.83          |        37.57          |      2.41     |
|      C      | GMFlow |    0.92     |   10.23     |         3.22          |         4.43          |        17.82          |        56.14          |      1.92     |
|     C+B     |  RAFT  |    0.89     |   11.64     |         2.58          |         5.17          |        14.01          |        47.53          |   **0.63**    |
|     C+B     | GMFlow |    1.34     |   12.23     |         3.93          |         5.27          |        22.83          |        64.41          |   **0.63**    |
|     C+T     |  RAFT  |    1.15     |    4.39     |         1.40          |         2.71          |         5.02          |        17.46          |      1.91     |
|     C+T     | GMFlow |    1.26     |    3.48     |         1.50          |         2.96          |        11.60          |        35.62          |      4.76     |
|    C+T+B    |  RAFT  |    1.24     |    6.45     |         1.54          |         2.82          |         7.35          |        20.27          |      0.70     |
|    C+T+B    | GMFlow |    1.46     |    4.04     |         1.66          |         3.11          |        14.50          |        44.35          |      0.67     |
|    C+T+S    |  RAFT  |    1.21     |    4.69     |         0.77          |         1.22          |         1.54          |         5.64          |      6.07     |
|    C+T+S    | GMFlow |    1.53     |    4.09     |         0.95          |         1.28          |         3.04          |        13.61          |      9.21     |
|   C+T+S+B   |  RAFT  |    1.35     |    6.49     |         0.87          |         1.48          |         1.79          |         6.31          |      0.64     |
|   C+T+S+B   | GMFlow |    1.66     |    4.43     |         1.04          |         1.42          |         3.99          |        18.83          |      0.65     |

### Pool Boiling Gravity

|    Model    | Method | Chairs(Val) | Things(Val) | Sintel(Train) - Clean | Sintel(Train) - Final | KITTI(Train) - F1-EPE | KITTI(Train) - F1-all | Boiling(Test) |
|-----------|------|-----------|-----------|---------------------|---------------------|---------------------|---------------------|-------------|
|      C      |  RAFT  |    0.82     |    9.03     |         2.19          |         4.49          |         9.83          |        37.57          |      3.30     |
|      C      | GMFlow |    0.92     |   10.23     |         3.22          |         4.43          |        17.82          |        56.14          |      2.40     |
|     C+B     |  RAFT  |    0.95     |   10.98     |         2.78          |         5.76          |        15.52          |        49.88          |   **0.90**    |
|     C+B     | GMFlow |    1.42     |   12.57     |         4.06          |         5.41          |        23.61          |        66.38          |   **0.90**    |
|     C+T     |  RAFT  |    1.15     |    4.39     |         1.40          |         2.71          |         5.02          |        17.46          |      2.75     |
|     C+T     | GMFlow |    1.26     |    3.48     |         1.50          |         2.96          |        11.60          |        35.62          |      3.60     |
|    C+T+B    |  RAFT  |    1.31     |    6.65     |         1.64          |         2.93          |         8.39          |        21.36          |      0.96     |
|    C+T+B    | GMFlow |    1.41     |    3.95     |         1.64          |         2.97          |        44.07          |        14.41          |      0.97     |
|    C+T+S    |  RAFT  |    1.21     |    4.69     |         0.77          |         1.22          |         1.54          |         5.64          |      4.74     |
|    C+T+S    | GMFlow |    1.53     |    4.09     |         0.95          |         1.28          |         3.04          |        13.61          |      5.49     |
|   C+T+S+B   |  RAFT  |    1.35     |    6.44     |         0.90          |         1.49          |         1.81          |         6.27          |      0.92     |
|   C+T+S+B   | GMFlow |    1.63     |    4.43     |         1.03          |         1.40          |         4.29          |        20.93          |      0.92     |

### Flow Boiling Inlet Velocity

|    Model    | Method | Chairs(Val) | Things(Val) | Sintel(Train) - Clean | Sintel(Train) - Final | KITTI(Train) - F1-EPE | KITTI(Train) - F1-all | Boiling(Test) |
|-----------|------|-----------|-----------|---------------------|---------------------|---------------------|---------------------|-------------|
|      C      |  RAFT  |    0.82     |    9.03     |         2.19          |         4.49          |         9.83          |        37.57          |     16.01     |
|      C      | GMFlow |    0.92     |   10.23     |         3.22          |         4.43          |        17.82          |        56.14          |     21.44     |
|     C+B     |  RAFT  |    1.21     |   14.99     |         3.62          |         6.78          |        22.07          |        60.64          |     10.13     |
|     C+B     | GMFlow |    1.50     |   13.96     |         4.52          |         5.89          |        23.45          |        65.96          |      7.01     |
|     C+T     |  RAFT  |    1.15     |    4.39     |         1.40          |         2.71          |         5.02          |        17.46          |     25.14     |
|     C+T     | GMFlow |    1.26     |    3.48     |         1.50          |         2.96          |        11.60          |        35.62          |     19.39     |
|    C+T+B    |  RAFT  |    1.62     |    9.09     |         2.19          |         3.77          |        13.81          |        32.13          |   **9.19**    |
|    C+T+B    | GMFlow |    1.45     |    4.15     |         1.78          |         3.05          |        15.74          |        48.34          |      7.24     |
|    C+T+S    |  RAFT  |    1.21     |    4.69     |         0.77          |         1.22          |         1.54          |         5.64          |     21.23     |
|    C+T+S    | GMFlow |    1.53     |    4.09     |         0.95          |         1.28          |         3.04          |        13.61          |     47.88     |
|   C+T+S+B   |  RAFT  |    2.18     |    9.32     |         1.54          |         2.90          |         2.57          |        10.93          |      9.68     |
|   C+T+S+B   | GMFlow |    1.72     |    4.80     |         1.18          |         1.65          |         4.69          |        24.56          |   **6.88**    |

### Flow Boiling Gravity

|    Model    | Method | Chairs(Val) | Things(Val) | Sintel(Train) - Clean | Sintel(Train) - Final | KITTI(Train) - F1-EPE | KITTI(Train) - F1-all | Boiling(Test) |
|-----------|------|-----------|-----------|---------------------|---------------------|---------------------|---------------------|-------------|
|      C      |  RAFT  |    0.82     |    9.03     |         2.19          |         4.49          |         9.83          |        37.57          |     20.42     |
|      C      | GMFlow |    0.92     |   10.23     |         3.22          |         4.43          |        17.82          |        56.14          |     14.05     |
|     C+B     |  RAFT  |    1.14     |   12.65     |         3.30          |         6.13          |        17.94          |        53.76          |      4.45     |
|     C+B     | GMFlow |    1.30     |   13.02     |         4.05          |         5.41          |        23.17          |        65.33          |      4.24     |
|     C+T     |  RAFT  |    1.15     |    4.39     |         1.40          |         2.71          |         5.02          |        17.46          |     21.04     |
|     C+T     | GMFlow |    1.26     |    3.48     |         1.50          |         2.96          |        11.60          |        35.62          |     13.37     |
|    C+T+B    |  RAFT  |    1.41     |    6.85     |         1.97          |         3.89          |        10.96          |        27.82          |      4.24     |
|    C+T+B    | GMFlow |    1.37     |    3.89     |         1.71          |         3.05          |        14.29          |        45.03          |      3.94     |
|    C+T+S    |  RAFT  |    1.21     |    4.69     |         0.77          |         1.22          |         1.54          |         5.64          |     21.04     |
|    C+T+S    | GMFlow |    1.53     |    4.09     |         0.95          |         1.28          |         3.04          |        13.61          |     22.84     |
|   C+T+S+B   |  RAFT  |    1.59     |    7.29     |         1.13          |         2.14          |         2.22          |         9.34          |   **4.12**    |
|   C+T+S+B   | GMFlow |    1.62     |    4.42     |         1.04          |         1.43          |         3.86          |        17.76          |   **3.93**    |


# SciML Models
## Model Checkpoints

The model checkpoints are organized based on the dataset they were trained on. So,
each link contains checkpoints for FNO, UNO and UNet. The remaining checkpoints will
be added soon.

| Models | Size |
|----|----|
| [Flow Boiling Gravity Models](https//bubbleml-model-checkpoints.s3.us-east-2.amazonaws.com/fb_gravity.tar.gz) | 237.0 MB |
| [Pool Boiling Saturated Models](https//bubbleml-model-checkpoints.s3.us-east-2.amazonaws.com/pb_saturated.tar.gz) | 208.9 MB |
| [Pool Boiling SubCooled Models](https//bubbleml-model-checkpoints.s3.us-east-2.amazonaws.com/pb_subcooled.tar.gz) | 208.9 MB |

## Model Benchmarks

### Pool Boiling Subcooled

|            | UNet-bench | UNet-mod | FNO   | UNO   | FFNO  | GFNO  |
|------------|-----------------------|--------------------|-------|-------|-------|-------|
| Rel. Err.  | **0.036**             | 0.051              | 0.052 | 0.062 | 0.050 | 0.071 |
| RMSE       | **0.035**             | 0.049              | 0.050 | 0.059 | 0.048 | 0.068 |
| BRMSE      | **0.073**             | 0.146              | 0.118 | 0.145 | 0.124 | 0.209 |
| IRMSE      | **0.113**             | 0.157              | 0.189 | 0.220 | 0.155 | 0.290 |
| Max Err.   | 2.1                   | **1.937**          | 2.32  | 2.659 | 2.21  | 2.882 |
| F. Low     | **0.281**             | 0.348              | 0.373 | 0.440 | 0.472 | 0.557 |
| F. Mid     | **0.266**             | 0.386              | 0.390 | 0.453 | 0.375 | 0.547 |
| F. High    | **0.040**             | 0.059              | 0.055 | 0.068 | 0.057 | 0.090 |

### Pool Boiling Saturated

|            | UNet-bench | UNet-mod | FNO   | UNO   | FFNO  | GFNO  |
|------------|-----------------------|--------------------|-------|-------|-------|-------|
| Rel. Err.  | **0.035**             | 0.039              | 0.052 | 0.072 | 0.053 | 0.066 |
| RMSE       | **0.035**             | 0.039              | 0.052 | 0.071 | 0.052 | 0.065 |
| BRMSE      | **0.082**             | 0.147              | 0.165 | 0.214 | 0.163 | 0.218 |
| IRMSE      | **0.052**             | 0.083              | 0.125 | 0.116 | 0.095 | 0.152 |
| Max Err.   | **1.701**             | 1.785              | 2.055 | 2.23  | 1.788 | 1.793 |
| F. Low     | 0.257                 | **0.241**          | 0.373 | 0.568 | 0.416 | 0.463 |
| F. Mid     | **0.268**             | 0.273              | 0.377 | 0.532 | 0.399 | 0.496 |
| F. High    | **0.023**             | 0.042              | 0.053 | 0.069 | 0.052 | 0.070 |


### Pool Boiling Gravity

|            | UNet-bench | UNet-mod | FNO   | UNO   | FFNO  | GFNO  |
|------------|-----------------------|--------------------|-------|-------|-------|-------|
| Rel. Err.  | **0.042**             | 0.051              | 0.062 | 0.081 | 0.061 | 0.124 |
| RMSE       | **0.040**             | 0.049              | 0.062 | 0.077 | 0.058 | 0.118 |
| BRMSE      | **0.070**             | 0.139              | 0.150 | 0.184 | 0.131 | 0.642 |
| IRMSE      | **0.108**             | 0.201              | 0.266 | 0.335 | 0.239 | 0.424 |
| Max Err.   | 2.92                  | **2.846**          | 3.626 | 3.667 | 3.22  | 4.0   |
| F. Low     | 0.491                 | **0.466**          | 0.500 | 0.746 | 0.578 | 1.095 |
| F. Mid     | **0.275**             | 0.326              | 0.440 | 0.548 | 0.423 | 0.760 |
| F. High    | **0.033**             | 0.049              | 0.054 | 0.077 | 0.049 | 0.171 |

### Flow Boiling Gravity

|            | UNet-bench | UNet-mod | FNO   | UNO   |
|------------|-----------------------|--------------------|-------|-------|
| Rel. Err.  | **0.055**             | 0.095              | 0.134 | 0.157 |
| RMSE       | **0.051**             | 0.088              | 0.123 | 0.144 |
| BRMSE      | **0.080**             | 0.240              | 0.267 | 0.301 |
| IRMSE      | **0.091**             | 0.214              | 0.351 | 0.379 |
| Max Err.   | **2.560**             | 2.800              | 3.951 | 3.990 |
| F. Low     | **0.263**             | 0.388              | 0.591 | 0.602 |
| F. Mid     | **0.281**             | 0.419              | 0.551 | 0.692 |
| F. High    | **0.149**             | 0.276              | 0.389 | 0.449 |


### Flow Boiling Inlet Velocity

|            | UNet-bench | UNet-mod | FNO   | UNO   |
|------------|-----------------------|--------------------|-------|-------|
| Rel. Err.  | **0.106**             | 0.122              | 0.202 | 0.225 |
| RMSE       | **0.097**             | 0.111              | 0.184 | 0.205 |
| BRMSE      | **0.271**             | 0.315              | 0.486 | 0.471 |
| IRMSE      | **0.178**             | 0.193              | 0.252 | 0.319 |
| Max Err.   | 2.900                 | **2.862**          | 3.527 | 3.992 |
| F. Low     | **0.571**             | 0.584              | 1.061 | 0.981 |
| F. Mid     | **0.511**             | 0.566              | 1.084 | 1.334 |
| F. High    | **0.264**             | 0.315              | 0.494 | 0.523 |

### Velocity + Temp: Pool Boiling Subcooled

|           | Metric   | UNet-bench | UNet-mod | P-UNet-mod | FNO   | UNO   | FFNO  |
|-----------|----------|-----------------------|--------------------|------------------------|-------|-------|-------|
| Temp.     | Rel. Err. | **0.106**             | 0.074              | **0.040**              | 0.066 | 0.322 | 0.067 |
|           | RMSE     | **0.097**             | 0.072              | **0.039**              | 0.064 | 0.051 | 0.065 |
|           | BRMSE    | **0.271**             | 0.211              | **0.113**              | 0.142 | 0.050 | 0.237 |
|           | IRMSE    | **0.178**             | 0.191              | **0.129**              | 0.212 | 0.077 | 0.201 |
|           | Max Err. | 2.83                  | **2.340**          | 2.741                  | 2.566 | 7.492 | 2.488 |
|           | F. Low   | **1.407**             | 0.495              | **0.244**              | 0.531 | 4.902 | 0.498 |
|           | F. Mid   | **0.740**             | 0.583              | **0.280**              | 0.518 | 1.594 | 0.507 |
|           | F. High  | **0.090**             | 0.076              | **0.051**              | 0.061 | 0.442 | 0.083 |
| $x$ Vel.  | Rel. Err. | **0.681**             | 0.553              | **0.429**              | 0.642 | 1.834 | 0.547 |
|           | RMSE     | **0.021**             | 0.015              | **0.012**              | 0.018 | 0.051 | 0.015 |
|           | BRMSE    | **0.023**             | 0.023              | **0.022**              | 0.023 | 0.050 | 0.024 |
|           | IRMSE    | **0.051**             | 0.049              | **0.043**              | 0.049 | 0.077 | 0.050 |
|           | Max Err. | **0.294**             | 0.320              | 0.353                  | 0.32  | 0.574 | 0.334 |
|           | F. Low   | **0.329**             | 0.225              | **0.142**              | 0.258 | 0.677 | 0.189 |
|           | F. Mid   | **0.115**             | 0.105              | **0.079**              | 0.110 | 0.328 | 0.107 |
|           | F. High  | **0.012**             | 0.012              | **0.011**              | 0.012 | 0.092 | 0.012 |
| $y$ Vel.  | Rel. Err. | **0.563**             | 0.400              | **0.311**              | 0.415 | 1.170 | 0.397 |
|           | RMSE     | **0.021**             | 0.015              | **0.012**              | 0.016 | 0.044 | 0.015 |
|           | BRMSE    | **0.015**             | 0.012              | **0.010**              | 0.011 | 0.035 | 0.013 |
|           | IRMSE    | **0.046**             | 0.046              | **0.038**              | 0.045 | 0.066 | 0.047 |
|           | Max Err. | **0.292**             | 0.272              | 0.280                  | 0.290 | 0.288 | 0.289 |
|           | F. Low   | **0.387**             | 0.190              | **0.132**              | 0.180 | 0.660 | 0.176 |
|           | F. Mid   | **0.128**             | 0.112              | **0.085**              | 0.111 | 0.240 | 0.106 |
|           | F. High  | **0.011**             | 0.010              | **0.009**              | 0.010 | 0.067 | 0.010 |

