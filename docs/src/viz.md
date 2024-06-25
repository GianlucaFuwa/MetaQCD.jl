# Visualization
We include the ability to visualize your data. For that, you have pass the the directory
under "ensembles" that contains your measurements, creating a `MetaMeasuremnts` object
holding all the measurements in `Dict` where the keys are symbols denoting the observable.
```julia
ens = "my_ensemble"
measurements = MetaMeasurements(ens)
```

Now we can plot a timeseries of any observables at flow time `tf` measured on the ensemble
via the
`timeseries` method:
```julia
timeseries(measurements, :myobservable, tf=0)
```

For hadron correlators there is a special function `hadroncorrelator` that plots the
mean values of all time slices (without statistical uncertainties). Just specify the hadron
whose correlator you want to see:
```julia
hadroncorrelator(measurements; logscale=true, calc_meff=false, tf=0.0)
```

You can also create a holder of a bias potential and plot it. MetaQCD.jl creates the bias
files with an extension that gives their type (.metad or .opes), but if you changed the
extension you have to provide the bias type as a symbol under the kwarg `which`:
```julia
bias = MetaBias(myfile, which=:mytype)
biaspotential(bias)
```
