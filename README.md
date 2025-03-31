# program_tai1_HSV_main

program_tai-main_for_Field_photo_diagnostic_imaging_HSV

(These programs were created with outstanding contributions by Shingo Tamachi and Shion Yamada master's degree in Chiba University)

Firstly, confirm the filename "Setup Instruction.txt" and "requirements.txt". The libraries and modules should be set up.

#HCV analysis

i) Commands for the evaluation within the range 

After performed the steps i)-iii) in the Texture analysis (./program_tai1_Texture),the command automatically compute the HSV values and statics values (python3 program.py; for Heu, program2h_17000_value.py or program2h_statics.py; for Satuation, program2s_17000_value.py or program2s_statics.py; for Value, program2v_17000_value.py or program2v_statics.py)

ii) Commands for the characteristic profile 

The Python command (library: scikit_posthocs etc.) automatically compute the difference of the profiling (python3 program.py)
