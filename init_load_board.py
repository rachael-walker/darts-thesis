import math
import numpy as np 

#---------------------------------------------------------------------------------------------
############################## ---- Main Board Dimensions ---- ###############################
#---------------------------------------------------------------------------------------------

# Initialize board radii 
r_do = 170
r_di = 162
r_to = 107
r_ti = 99
r_sb = 15.9 
r_db = 6.35

# Regions are related to the radii on the board
# Have removed the bullseyes since they are special cases
board_regions = {'SI':1,'SO':1,'D':2,'T':3}

# Segments are related to the angle on the board 
board_segments = [20, 1, 18 , 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# Each board segment will have three orientations : 
#   - "c" for centre (along the centre of the segment)
#   - "cc" for counter clockwise (half-way beteween the middle of the segment and the edge of the segment in the counterclockwise direction)
#   - "cw" for clockwise (half-way beteween the middle of the segment and the edge of the segment in the clockwise direction)
# Bullseyses are special case and have been split out.

theta_m = [math.radians(18*i) for i in range(0,20)]
theta_cc = [math.radians(18*i - 4.5) for i in range(0,20)]
theta_cw = [math.radians(18*i + 4.5) for i in range(0,20)]

board_segment_orientations = {'c':theta_m,'cc':theta_cc,'cw':theta_cw}
board_segment_orientations_bullseyes = {'n':math.radians(0),'s':math.radians(180),'e':math.radians(90),'w':math.radians(270),
                                        'ne':math.radians(45),'se':math.radians(135),'sw':math.radians(225),'nw':math.radians(315)}

# Each board region will have three radial distances: 
#   - "o" for outer where the radius is halfway between the centre of the region and the outer border
#   - "m" for middle where the radius is in the centre of the region 
#   - "i" for inner where the radius is halfway between the centre of the region and the inner border 

board_region_distances = {}

#---------------------------------------------------------------------------------------------
################################# ---- Calculate Radii ---- ##################################
#---------------------------------------------------------------------------------------------

# Triples target radii
r_t_outer = (r_to + r_ti)/2 + (r_to - r_ti)/4
r_t_middle = (r_to + r_ti)/2
r_t_inner = (r_to + r_ti)/2 - (r_to - r_ti)/4
board_region_distances['T'] = {'o':r_t_outer,'m':r_t_middle,'i':r_t_inner}

# Doubles target radii
r_d_outer = (r_do + r_di)/2 + (r_do - r_di)/4
r_d_middle = (r_do + r_di)/2
r_d_inner = (r_do + r_di)/2 - + (r_do - r_di)/4
board_region_distances['D'] = {'o':r_d_outer,'m':r_d_middle,'i':r_d_inner}

# Singles outer target radii
r_so_outer = (r_di + r_to)/2 + (r_di - r_to)/4
r_so_middle = (r_di + r_to)/2
r_so_inner = (r_di + r_to)/2 - (r_di - r_to)/4
board_region_distances['SO'] = {'o':r_so_outer,'m':r_so_middle,'i':r_so_inner}

# Singles inner target radii
r_si_outer = (r_ti + r_sb)/2 + (r_ti - r_sb)/4
r_si_middle = (r_ti + r_sb)/2
r_si_inner = (r_ti + r_sb)/2 - (r_ti - r_sb)/4
board_region_distances['SI'] = {'o':r_si_outer,'m':r_si_middle,'i':r_si_inner}

# Bullseyes
r_sb_middle = (r_sb + r_db) / 2
board_region_distances['SB'] = {'m':r_sb_middle}

r_db_middle = 0 
board_region_distances['DB'] = {'m':r_db_middle}