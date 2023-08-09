import math
import numpy as np 
from matplotlib import pyplot as plt

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


# Source https://www.pythonpool.com/matplotlib-circle/

def plot_basic_board(figsize = (8,8),xlim=(-200,200),ylim=(-200,200),axes=None):

  plt.rcParams["figure.figsize"] = figsize
  
  # list of angles for plotting circles
  angle = np.linspace( 0 , 2 * np.pi , 150 ) 

  # outer doubles circle  
  r_do = 170
  x_do = r_do * np.cos( angle ) 
  y_do = r_do * np.sin( angle ) 

  # inner doubles circle
  r_di = 162
  x_di = r_di * np.cos( angle ) 
  y_di = r_di * np.sin( angle ) 

  # outer triples circle
  r_to = 107
  x_to = r_to * np.cos( angle ) 
  y_to = r_to * np.sin( angle ) 

  # inner triples circle
  r_ti = 99
  x_ti = r_ti * np.cos( angle ) 
  y_ti = r_ti * np.sin( angle ) 

  # single bullseye circle
  r_sb = 15.9  
  x_sb = r_sb * np.cos( angle ) 
  y_sb = r_sb * np.sin( angle ) 

  # double bullseye circle
  r_db = 6.35
  x_db = r_db * np.cos( angle ) 
  y_db = r_db * np.sin( angle ) 

  # radial line list 
  x_line_list = []
  y_line_list = []
  for i in range(0,20):
    x_line_list.append([r_sb*np.sin(math.radians(9+18*i)),r_do*np.sin(math.radians(9+18*i))])
    y_line_list.append([r_sb*np.cos(math.radians(9+18*i)),r_do*np.cos(math.radians(9+18*i))])

  # initialize plot
  #figure, axes = plt.subplots( 1 ) 

  if axes == None:
        axes = plt.gca()
  
  # plot circles
  axes.plot( x_do, y_do , 'k') 
  axes.plot( x_di, y_di , 'k') 
  axes.plot( x_to, y_to , 'k') 
  axes.plot( x_ti, y_ti , 'k') 
  axes.plot( x_sb, y_sb , 'k') 
  axes.plot( x_db, y_db , 'k') 
    
  axes.set_aspect( 1 ) 

  # plot lines
  segment_vals = [20, 1, 18 , 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

  for i in range(0,20):
    
    # plot line 
    axes.plot(x_line_list[i], y_line_list[i], 'k')   
    
    # plot label 
    coords = ((r_do*1.1)*np.sin(math.radians(18*i)), (r_do*1.1)*np.cos(math.radians(18*i)))                                 
    axes.annotate('%d' % segment_vals[i], xy=coords, textcoords='data',ha='center',va='center') 

  # plt.xlim(-200,200)
  # plt.ylim(-200,200)
  axes.set_xlim(xlim)
  axes.set_ylim(ylim)

  axes.tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False) # labels along the bottom edge are off
  axes.tick_params(
      axis='y',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      left=False,      # ticks along the bottom edge are off
      right=False,         # ticks along the top edge are off
      labelleft=False) # labels along the bottom edge are off

  return axes