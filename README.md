# Project 1: Intro to Locally Weighted Regression

#### Main Ideas

The main idea of linear regression is to make predictions as a weighted combination of the input feauture values where the weights can be positive or negative. As an equation, linear regression looks like:

$$ \text{Predicted Value} = weight_1 \cdot \text{Feature}_1 + weight_2 \cdot \text{Feature}_2 + ... + weight_p \cdot \text{Feature}_p $$

<script src="http://latex.codecogs.com/latexit.js"></script>
<div lang="latex">
	\frac{1+sin(x)}{y}
</div>

Locally weighted regression is a method for computing a non-linear trend line for data. This ability is important, as many real-life associations and trends are not linear. Locally weighted regression uses a kernel to calculate a linear regression for the neighborhood each data point, that, when stitched together, show the overall (often non-linear) trend for the data. So, even though trends and associations are generally nonlinear, they can be locally interpreted linearly.

Local properties are relative to a metric, which is a method by which we compute the distance between two observations. Observations contain multiple features, and if they are numeric, we can see them as vectors. The equation for calculating the distance between two p-dimensional vectors is as follows:

$$ dist(\vec{v},\vec{w})=\sqrt{(v_1-w_1)^2+(v_2-w_2)^2+...(v_p-w_p)^2}$$

In the end, you have the same number of weight vectors as there are different observations in the data.

#### Kernels

When you are preparing to run a locally weighted regression, there are first some hyperparameters you must choose. One of these hyperparameters is the kernel, which determines the shape of the bell-curve that is used to calculate the local weights. This kernel is centered on the current observation, giving more weight to other data points that are closer to the current observation. A visualization is provided below.

<img src="Loess_1.drawio.svg" width="400" height="300" alt="hi" class="inline"/>

Four possible kernel options are the Exponential, the Tricubic, the Epanechnikov, and the Quartic. You can see by looking at the two plots below that each kernel is slightly different. For example, the Tricubic Kernel (left) has thinner tails and a flatter top than the Epanechnikov (right).

<img src="tricubic.png" width="400" height="300" alt="hi" class="inline"/> <img src="epanechnikov.png" width="400" height="300" alt="hi" class="inline"/>
