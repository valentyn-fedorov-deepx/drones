Script will check all .pxi files in the pxi_sources folder and save outputs to the normals_outputs folder.

First you need to run setup.bat that will create virtual environment and instasll dependencies. This should be done only once at the beginning. For running script itself there is a run.bat file.  

Outputs:
- i0.png - from VyzAPI
- i45.png - from VyzAPI
- i90.png - from VyzAPI
- i135.png - from VyzAPI
- s0.png - from VyzAPI
- s1.png - from VyzAPI
- s2.png - from VyzAPI
- raw.png - raw 
- s0_python.png - calculated in python using i0, i45, i90, and i135 from VyzAPI
- s1_python.png - calculated in python using i0, i45, i90, and i135 from VyzAPI
- s2_python.png - calculated in python using i0, i45, i90, and i135 from VyzAPI
- n_z.png - normals from s0, s1, and s2 calculated in python
- n_xyz.png - normals from s0, s1, and s2 calculated in python
- n_xy.png - normals from s0, s1, and s2 calculated in python
- u_i.png - normals from s0, s1, and s2 calculated in python
- z.png - fromVyzAPI
- xy.png - fromVyzAPI
- xyz.png - fromVyzAPI
- current_people-track_u_i.png - outputs like in the current people-track application
- current_people-track_nz.png - outputs like in the current people-track application
- current_people-track_nxy.png - outputs like in the current people-track application
- current_people-track_nxyz.png - outputs like in the current people-track application
- s1.csv - raw values from vyzapi
- s2.csv - raw values from vyzapi
- z.csv - raw values from vyzapi
- xy.csv - raw values from vyzapi
- xyz_0.csv - raw values from vyzapi for the first channel
- xyz_1.csv - raw values from vyzapi for the second channel
- xyz_2.csv - raw values from vyzapi for the third channel
- diff_z.png - difference between people-track and shapeos normals
- diff_xy.png - difference between people-track and shapeos normals
- diff_xyz.png - difference between people-track and shapeos normals