# Project 1: Intro to Locally Weighted Regression

The main idea of linear regression is to make predictions as a weighted combination of the input feauture values where the weights can be positive or negative. As an equation, linear regression looks like:

$$\text{Predicted Value} = weight_1 \cdot \text{Feature}_1 + weight_2 \cdot \text{Feature}_2 + ... + weight_p \cdot \text{Feature}_p $$

Locally weighted regression is a method for computing a non-linear trend line for data. This ability is important, as many real-life associations and trends are not linear. Locally weighted regression uses a kernel to calculate a linear regression for the neighborhood each data point, that, when stitched together, show the overall (often non-linear) trend for the data. So, even though trends and associations are generally nonlinear, they can be locally interpreted linearly.

Local properties are relative to a metric, which is a method by which we compute the distance between two observations. Observations contain multiple features, and if they are numeric, we can see them as vectors. The equation for calculating the distance between two p-dimensional vectors is as follows:

$$ dist(\vec{v},\vec{w})=\sqrt{(v_1-w_1)^2+(v_2-w_2)^2+...(v_p-w_p)^2}$$

In the end, you have the same number of weight vectors as there are different observations in the data.
