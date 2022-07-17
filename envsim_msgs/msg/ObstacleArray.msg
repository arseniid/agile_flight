Header header 

# Time [s]
float64 t

# Number of obstacles in message
int64 num_obstacles

# Number of free paths in message
int64 num_free_paths

# Obstacle array
envsim_msgs/Obstacle[] obstacles

# Polar voxel array
envsim_msgs/PolarVoxel[] free_paths
