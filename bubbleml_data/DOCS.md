# BubbleML Documentation

The BubbleML dataset consists of several studies, each composed of multiple simulations. 
Each of these simulations is stored as one HDF5 file. All of the HDF5 files store relevent tensor data:

1. temperature `temperature`
2. pressure gradient `pressure`
3. x-velocity `velx`
4. y-velocity `vely`
5. signed distance function `dfun`
6. x-coordinate grid `x`
7. y-coordinate grid `y`
8. real-valud runtime parameters `real-runtime-params`
9. integer-valued runtime paramters `int-runtime-params`

The simulation data can be accessed using h5py. Here, we load the temperature data into a torch tensor.

```python
import h5py
import torch

with h5py.File(<path-to-sim>) as f:
    temp = torch.from_numpy(f['temp'][:])
    real_params = f['real-runtime-params'][:]
```

All simulations fields are laid out in memory identically: `T x X x Y`. This layout makes indexing hdf5 files by time faster
since each domain will be laid out contiguously in memory. In our experiments, we always index by time. Every field will have
an identical shape:

```python
f['temperature'][:].shape == f['pressure'][:].shape == ...
``` 

For a full example of how to read and visualize each field, check [the data loading example](../examples/data_loading.ipynb).

## Metadata (runtime-params)

There is a lot of metadata associated with each of the simulations. Some settings may be difficult to interpret and most will be unnecessary. We list out keys that are particularly relevant. Some of these settings, like the Reynolds and Prandtl number are important parameters used for the governing equation. These will be critical when implementing a physics-informed model.

Real runtime parameters (`real-runtime-params`):
1. Inverse Reynold's number: `ins_invreynolds`
2. Stefan Number: `mph_stefan`
3. Prandtl Number: `ht_prandtl`
4. Non-dimenionalized Saturation temperature: `mph_tsat` 
5. Non-dimenionalized Bulk temperature: `ht_tbulk` 
6. Non-dimensional min and max temperature: `ht_twall_high`, `ht_twall_low`
7. Domain sizes: `xmin`, `xmax`, `ymin`, `ymax`

The governing equation for the vapor phase includes the [thermal diffusivity](https://en.wikipedia.org/wiki/Thermal_diffusivity). This is computed from three values:
1. Specific heat capacity: `mph_cpgas`,
2. Density: `mph_rhogas`,
3. Thermal Conductivity: `mph_thcogas`

In the liquid phase, the thermal diffusivity is set to one.

The integer runtime parameters include settings for the resolution and are necessary for unblocking. These may be necessary to use if trying to extend BubbleML and want to unblock the dataset. 
Integer runtime parameters (`int-runtime-params`):
1. The number of blocks in the x,y,z directions: `nblockx`, `nblocky`, `nblockz`
2. The block sizes in the x,y,z directions: `gr_tilesizex`, `gr_tilesizex`, `gr_tilesizex` 

The resolution in the x-direction can be computed using `nblockx * gr_tilesizex`. An example using the integer runtime parameters to unblock a dataset can be seen in our [scripts](../scripts/boxkit_dataset.py). These integer settings are read by [boxkit](https://github.com/akashdhruv/BoxKit) to simplify reconstructing a simulation.

## Temperature

The temperature is stored in a non-dimensionalized form. This means that the stored temperature will always range from [0-1].
In studies where we vary the heater temperature, the temperature ranges from the liquid temperature, to the heater temperature.
So, in each case, the heater tempeature is normalized to 1. This requires re-dimensionalizing the data. This is simple and can just be
done by multiplting the non-dimensionalized temperature by the heater temperature:

```python
heater_temp = get_heater_temp(sim_file)
temp *= heater_temp
```

Once this is done, it should be safe to use. In the studies where the heater temperature is constant (i.e., where we vary the gravity
or inlet velocity), this is unnecessary.

## Pressure

Each of the simulation files stores the pressure gradient, not the actual pressure. This is because only the pressure gradient is used
in the governing equations. The pressure is computed by solving a Poisson equation. We have noticed that the Poisson solver may not be
sufficiently robust to be used on its own. In the numerical simulations, this is fine because its main purpose is to correct the velocities, not
serve as a truly accurate model of pressure. In our experiments, we did not use the pressure, but we make note of it for future users who may be interested. 
It would be interesting to incorporate the pressure into models and test whether velocity predictions improve.

## Distance function

Each simulation includes a field `dfun`, which is a signed distance function to the nearest bubble.
When a point is in the vapor phase, `dfun > 0`. When a point is in the liquid phase, `dfun <= 0`. This field can
be used to get a mask for all liquid points, all vapor points, or points along the bubble interface. In the [example](../examples/data_loading.ipynb),
we include an example of how to compute the liquid-vapor interface using the same heavy-side function as the simulation.
We use the distance function to generate a mask of bubble locations (I.e., points in the vapor phase.)

## The Domain

The simulation data we provide does not include the boundary. For instance,
`f['temperature'][:, 0, 0]` is not indexing the heater. Instead, it is indexing the cell just above the heater. Similarly, 
`f['temperature'][:, 0, 10]` is not indexing the wall, it is indexing the cell to the right of the wall. If you want to explicitly
account for boundaries in your model (perhaps for a physics-informed neural network), you must handle it implicitly, or extend the domain
with the boundary info. In our experiments, we treat it implicitly and assume that the model will be able to capture the boundary info
from the input history. 

## Extending BubbleML

We provide a [reproducibility capsule](https://github.com/Lab-Notebooks/Outflow-Forcing-BubbleML) for running the simulations with Flash-X. This
includes lab notebooks for running simulations. It also includes analysis scripts and the submissions files used to generate BubbleML.
