## Multiple Variable Linear Regression
In this project, we will implement the data structures and routines to support linear regression for multiple features.
#### Goals
- Implement our regression model routines to support multiple features
    - Implement data structures to support multiple features
    - Write prediction, cost and gradient routines to support multiple features
    - Utilize NumPy np.dot to vectorize their implementations for speed and simplicity
#### Tools
In this project, we will make use of:
- NumPy, a popular library for scientific computing
- Matplotlib, a popular library for plotting data
```python
import copy, math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays
```
#### Notation
Here is a summary of some of the notation we will encounter, updated for multiple features.
| **General Notation** | **Description** | **Python (If applicable)** |
|----------------------|-----------------|----------------------------|
| *a* | scalar, non bold | |
| **a** | vector, bold | |
| **A** | matrix, bold capital | |
| **Regression** | | |
| **X** | training example matrix | X_train |
| **y** | training example targets | y_train |
| **x**<sup>(i)</sup>, ***y***<sup>(i)</sup> | ***i***<sub>th</sub>Training Example | X[i], y[i] |
| m | number of training example | `m` |
| n | number of training example | `n` |
| **w** | parameter: weight, | `w` |
| *b* | parameter: bias | `b` |
| *𝑓*<sub>𝐰,𝑏</sub>(**𝐱**<sup>(𝑖)</sup>) | The result of the model evaluation at  𝐱<sup>(𝐢)</sup> parameterized by 𝐰,𝑏: *𝑓*<sub>𝐰,𝑏</sub>(𝐱<sup>(𝑖)</sup>)=𝐰⋅𝐱(<sup>(𝑖)</sup>)+𝑏 | `f_wb` |
#### <a name="toc_15456_2"></a>
# Problem Statement
We will use the housing price prediction example. The training dataset contains three examples with four features (size, bedrooms, floors and, age) shown in the table below.  Note the size is in sqft rather than 1000 sqft.
| Size (sqft) | Number of Bedrooms  | Number of floors | Age of  Home | Price (1000s dollars)  |   
| ----------------| ------------------- |----------------- |--------------|-------------- |  
| 2104            | 5                   | 1                | 45           | 460           |  
| 1416            | 3                   | 2                | 40           | 232           |  
| 852             | 2                   | 1                | 35           | 178           |  
We will build a linear regression model using these values so we can then predict the price for other houses. For example, a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old.  
Let's run the following code cell to create your `X_train` and `y_train` variables.
