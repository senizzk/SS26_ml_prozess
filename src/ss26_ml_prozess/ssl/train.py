import lejepa

# Choose a univariate test (Epps-Pulley in this example)
univariate_test = lejepa.univariate.EppsPulley(n_points=17)

# Create the multivariate slicing test
loss_fn = lejepa.multivariate.SlicingUnivariateTest(
    univariate_test=univariate_test, num_slices=1024
)
