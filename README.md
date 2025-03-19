# generalization_error

![relaxation](https://github.com/user-attachments/assets/1b2bb3e1-2d87-4edf-8055-6783aa199539)

Step 1.

Generate 10000 sets of weights (1284 parameters each).

$$
w \sim \mathcal{N}(0, \sigma^2)
$$


Step 2.

Calculate the mean for each set.  
Therefore, there are 10000 means.


Step 3.

Transfer the means to P(c) based on

$$
P(c) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(c - 0)^2}{2\sigma^2} \right)
$$


Step 4.

Unify the probability.

![MC](https://github.com/user-attachments/assets/ff006fa5-7bf6-45f4-b9d5-2f4b6039bb9e)



-----
-----


Step 1.

Load the weight.


Step 2.

Assume the distribution. N(0, sigma^2)


Step 3.

Calculate the probability of this combination.
