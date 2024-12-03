# This script postprocesses raw data from experimental campaigns conducted on a composite material 
# made of lunar regolith simulant (EAC-1a) and PEEK. 
# The objective is to analyze mechanical behavior based on data from compression and bending tests. 
#
# A full-factorial L12 Design of Experiments (DoE) approach was applied to account for variations 
# in processing parameters during manufacturing. Each combination of parameters was tested using 
# six identical specimens. 
#
# Data format:
# - Test results: Filenames follow the pattern BDoE00.txt (bending) and CDoE00.txt (compression), 
#   where "00" represents the specimen number. Columns include: 'Time', 'Displ', 'Load'.
# - Specimen dimensions: Filenames BDoE_dim.txt (bending) and CDoE_dim.txt (compression). 
#   Columns: 'depth', 'width' (bending), 'diameter', 'height' (compression).
#
# Calculations:
# - Strength: Calculated at the maximum load sustained by each specimen.
# - Modulus: An adaptive algorithm identifies the linear region of the stress-strain curve 
#   using rolling linear regression. This method dynamically adjusts to variations in curve trends 
#   and ensures robustness by requiring slope consistency (variation ≤10%) across overlapping windows.
#
# V1.0 - 03/12/2024 - Roberto Torre
# Spaceship EAC - European Astronaut Centre / European Space Agency, Cologne, DE
