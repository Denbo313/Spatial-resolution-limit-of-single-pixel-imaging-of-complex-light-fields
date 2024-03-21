# Spatial-resolution-limit-of-single-pixel-imaging-of-complex-light-fields
Supplementary data and code supporting the article 'Spatial resolution limit of single pixel imaging of complex light fields'

The following files and folders are in this directory:

1) 'data_fields': Contains the experimental and simulated data stored as .CSV files for each data set individually.
2) 'field_reconstruction': Contains the images of the reconstructed amplitude and phase distributions.
3) 'input_fields': Contains the Gaussian and Dog amplitudes and the Boat phase as individual .CSV files.
4) 'simulate_experiment.py': Runs the simulation of the experiment
5) 'evaluate_csv_data.py': Reads and evaluates the experimental and simulated data and calculates the NCC values.
The NCC values are calculated and stored in properly named variables. Finally, they are saved in the file 'NCC_results.csv' 
The entries are ordered according to N = 64, 256, 1024.


##############################################
!!!IMPORTANT!!! Right now the folder 'field_reconstruction' is EMPTY! 
In order to create the data, change LINE 88 in 'evaluate_csv_data.py' to:
88 export_data = True

After runing the code, .png and .eps files will be created in the folder 'field_reconstruction'
############################################## 

For any questions, please contact: dennis.scheidt@correo.nucleares.unam.mx
