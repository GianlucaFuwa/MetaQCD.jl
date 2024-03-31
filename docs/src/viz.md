# Visualization:
We include the ability to visualize your data. For that, you have to activate and instantiate the visualization project:
```julia
pkg> activate ./visualize
pkg> instantiate
```

Now you can create a holder for all measurements in a directory and plot a time series of an observable, specifying its filename (without extenstion) as a symbol:
```julia
measurements = MetaMeasurements(mydir)
timeseries(measurements, :myobservable)
```

You can also create a holder of a bias potential and plot it. MetaQCD.jl creates the bias files with an extension that gives their type (.metad or .opes), but if you changed the extension you have to provide the bias type as a symbol under the kwarg `which`:
```julia
bias = MetaBias(myfile, which=:mytype)
biaspotential(bias)
```