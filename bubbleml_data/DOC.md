# BubbleML Documentation

The BubbleML dataset consists of several studies, each composed of multiple simulations. 
Each of these simulations is stored as one HDF5 file. All of the HDF5 files store relevent tensor data:

1. temperature `'temperature'`
2. pressure gradient `'pressure'`
3. x-velocity `velx`
4. y-velocity `vely`
5. signed distance function `dfun`
6. x-coordinate grid `x`
7. y-coordinate grid `y`
8. real-valud runtime parameters `real-runtime-params`
9. integer-valued runtime paramters `int-runtime-params`

The fields field can be accessed using h5py. Here, we load the temperature data into a torch tensor.

```python
import h5py
import torch

with h5py.File(<path-to-sim>) as f:
    temp = torch.from_numpy(f['temp'][:])
    real_params = f['real-runtime-params'][:]
```

All simulations are laid out in memory identically: `T x X x Y`. This layout makes indexing hdf5 files by time faster
since each domain will be laid out contiguously in memory. In our experiments, we always index by time.  

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

In the liquid phase, the thermal diffusivity is just set to one.

The integer runtime parameters includes settings for the resolution. These may be necessary to use if trying to extend BubbleML and want to unblock the dataset. 
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
