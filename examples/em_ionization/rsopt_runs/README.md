Launches an ensemble of simulations using ```rspot```.

rspot will either take a file containing the grid points to run or sample points within specified bounds.
To create training data do:
```ruby 
     $ python sampler.py
  ```  
Simulations can be run on serial or distributed across mpi nodes.

Executes with:
```ruby 
     $ source run.sh
  ```   
and may take anywhere from a few minutes to a few days, depending on the number of simulations needed and filesystem.

Relevant information is printed to stdout and dumped into several files in the ```user_scan_data/worker*```.
