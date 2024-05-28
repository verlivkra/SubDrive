#! python 3
# -*- coding: utf-8 -*-
"""
Created on 4/9/2024
@author: veronlk

Miscellaneous tools. 
"""
# --- MODULE IMPORTS ---------------------------------------------------------#
import numpy as np
import yaml
import os
import shutil
from openfast_toolbox.io  import FASTInputFile
import scipy.integrate as integrate

# --- FUNCTION DEFINITIONS ---------------------------------------------------#

def unique_val_list(x):
    """Function that checks that list contains unique values"""
    if len(x) > len(set(x)):
        return False
    else:
        return True


def cosd(deg): 
    return np.cos(np.deg2rad(deg))

def sind(deg): 
    return np.sin(np.deg2rad(deg))

def tand(deg): 
    return np.tan(np.deg2rad(deg))

def quadratic_equation(a, b, c):
    
    x1 = (-b+np.sqrt(b**2-4*a*c))/(2*a)
    x2 = (-b-np.sqrt(b**2-4*a*c))/(2*a)
    
    return x1, x2

def pipe_area(OD, t):
    IR  = (OD-2*t)/2
    OR  = OD/2
    A   = np.pi*(OR**2-IR**2) 
    return A

class TaperedPipe():
    """Calculate beam properties of tapered pipe member. 
        Assumes constant material density along the member, and linear variation of diameter and thickness."""
    def __init__(self, rho, OD1, OD2, t1, t2, coord1, coord2, loc_y_parallel_2_glob_y, TTZ = 0):
        
        # NB! TTZ only used for testing
        self.x1, self.y1, self.z1 = coord1[0], coord1[1], coord1[2] - TTZ 
        if TTZ != 0:
            print("WARNING! REMOVE TOWER HEIGHT FROM TAPEREDPIPE CALCULATION. THIS IS ONLY FOR CHECKING!")
        self.x2, self.y2, self.z2 = coord2[0], coord2[1], coord2[2] - TTZ
        
        self.L = np.sqrt((self.x1-self.x2)**2+(self.y1-self.y2)**2+(self.z1-self.z2)**2)
        self.rho = rho
        self.OD1 = OD1
        self.OD2 = OD2
        self.t1  = t1
        self.t2  = t2

        self.ID1 = self.OD1 - 2*self.t1
        self.ID2 = self.OD2 - 2*self.t2
        self.loc_y_parallel_2_glob_y = loc_y_parallel_2_glob_y
        self.areas()
        self.volume()
        self.calc_mass()
        
        self.vector()
        
        self.local_COM()
        self.local_COM_midlength()
        
        self.global_COM()
        self.global_COM_midlength()

        self.MoI_locCoord_midLength()
        self.MoI_globCoord_midLength()

    def areas(self):
        self.A1 = pipe_area(self.OD1, self.t1)
        self.A2 = pipe_area(self.OD2, self.t2)
    
    def volume(self):
        self.V = self.L/3*(self.A1 + np.sqrt(self.A1*self.A2) + self.A2)

    def calc_mass(self):
        self.mass = self.V*self.rho
    
    def vector(self):
        # Normalized orientation vector
        v = np.array([self.x2 - self.x1, self.y2 - self.y1, self.z2 - self.z1])
        v_mag = np.linalg.norm(v)
        self.v_norm = v / v_mag

    def local_COM(self):
        self.CMx_local = (self.A1 * self.L**2 / 2 + self.L**2 * (self.A2 - self.A1) / 3) / (self.A1 * self.L + self.L * (self.A2 - self.A1) / 2) # Along longitudinal axis of member
        

    def local_COM_midlength(self):
        self.CMx_local_midlength = self.L/2

    def global_COM(self):
        self.CMx = self.x1 + self.CMx_local * self.v_norm[0]
        self.CMy = self.y1 + self.CMx_local * self.v_norm[1]
        self.CMz = self.z1 + self.CMx_local * self.v_norm[2]
    
    def global_COM_midlength(self):
        self.CMx_global_midlength = self.x1 + self.CMx_local_midlength * self.v_norm[0]
        self.CMy_global_midlength = self.y1 + self.CMx_local_midlength * self.v_norm[1]
        self.CMz_global_midlength = self.z1 + self.CMx_local_midlength * self.v_norm[2]

    def MoI_locCoord_midLength(self):
        """Mass moment of inertia in local coordinate system relative to mid length"""
        # TODO! Using the mean of the diameter and thickness to calculate moment of inertia - hence "nontapered"
        # NB! Calculations based on mass center halfway along the length
        # x-axis along the beam

        self.ORmean = (self.OD1 + self.OD2)/2/2 # Average radius
        self.IRmean = (self.ID1 + self.ID2)/2/2 
        self.I_local_midlength = np.diag([
                1/2*self.mass*(self.ORmean**2 + self.IRmean**2), 
                1/12*self.mass*(3*(self.ORmean**2 + self.IRmean**2) + self.L**2),
                1/12*self.mass*(3*(self.ORmean**2 + self.IRmean**2) + self.L**2),
            ])

    def MoI_globCoord_midLength(self):
        """Mass moment of inertia in global coordinate system relative to mid length, shifted to correct center of mass for tapered beams"""
        # Calculating the moment of inertia based using the averaged diameter and thickness and assuming a non-tapered cylinder (simplification)

        if self.loc_y_parallel_2_glob_y != True: 
            print('Warning! When calculating the global moment if inertia, we assume that the local and global y-axis are parallel (but may not point in the same direction).')
            print('Note also that the local x-axis is assumed parallell to the longitudinal direction of the member.')
            print('If this member is involved in a gearbox support, this assumption is not necessarily valid.')
            # TODO: Update for gearbox torque arms
        
        local_x = self.v_norm
        local_y = np.array([0, 1, 0])
        local_z = np.cross(local_x, local_y)

        if np.linalg.norm(local_z) == 0:  # This happens if local_x is parallel or anti-parallel to local_y
            local_z = np.array([0, 0, 1])  # Choose global Z if degenerate case
            print('warning...')
        else:
            local_z /= np.linalg.norm(local_z)
        
        # Correct local_y to ensure a right-handed coordinate system
        local_y = np.cross(local_z, local_x)

        # Assemble rotation matrix from local to global
        R = np.vstack([local_x, local_y, local_z]).T

        # Steiner's theorem to find moments about correct (tapered) beam, not midlength
        distance = np.array([self.CMx - self.CMx_global_midlength, self.CMy - self.CMy_global_midlength, self.CMz - self.CMz_global_midlength])
        d_squared = np.dot(distance, distance)
        
        I_steiner = self.mass * d_squared * np.eye(3) - self.mass * np.outer(distance, distance)
        self.I_global_midlength = R @ (self.I_local_midlength) @ R.T   # CM midlength
        self.I_global_tapered = self.I_global_midlength + I_steiner     # Relative to CM for tapered beam 

        # print("Rotation Matrix (Local to Global):")
        # print(R)
        # print("Local Moments of Inertia about the Local Origin (midlength):")
        # print(self.I_local_midlength)

        # print("Global Moments of Inertia about the local Origin (midlength):")
        # print(self.I_global_midlength)

        # print("Global Moments of Inertia about the center of mass of tapered pipe, but calculated assuming straight pipe:")
        # print(self.I_global_tapered)
        self.JMXX = np.diag(self.I_global_tapered)[0]
        self.JMYY = np.diag(self.I_global_tapered)[1]
        self.JMZZ = np.diag(self.I_global_tapered)[2]
    

# --- Read stuff ---------------------------------------------------#

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        # Load the YAML content
        data = yaml.safe_load(file)
    return data

def add_quotationmarks(str_to_enclose):
    """Encloses a string with quotation marks. Useful when referring to files in OpenFAST-files. 
        Example: The string 'DTU_10MW_RWT_SubDyn.dat' becomes '"DTU_10MW_RWT_SubDyn.dat"'
    """
    return f'"{str_to_enclose}"'


# --- Folder and file handling ---------------------------------------------------#
def find_folder_by_keyword(directory, keyword):
    """Return the first folder within `directory` that contains `keyword` in its name,
    and provide feedback about found folders and warnings if multiple are found."""
    try:
        # List to store all directories containing the keyword
        matching_folders = []
        
        # Iterate over each item in the directory
        with os.scandir(directory) as entries:
            for entry in entries:
                # Check if the entry is a directory and if the keyword is in the directory name
                if entry.is_dir() and keyword in entry.name:
                    matching_folders.append(entry.name)
        
        # Check if there are multiple matching folders
        if len(matching_folders) > 1:
            print("Warning: Multiple folders contain the keyword. Choosing the first one found.")
            for folder in matching_folders:
                print(f"Matching folder: {folder}")
        
        # Return the first matching folder or None if no match is found
        if matching_folders:
            #print(f"Found and selected folder by keyword {keyword}: {matching_folders[0]} ")
            return matching_folders[0]
        else:
            print("No folder found with that keyword.")
            return None

    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
    except PermissionError:
        print(f"Permission denied while accessing {directory}.")

def find_file_by_extension(directory, extension):
    """Return the first file within `directory` that has `extension` as its file extension,
    and provide feedback about found files and warnings if multiple are found."""
    try:
        # List to store all directories containing the keyword
        matching_files = []
        
        # Iterate over each item in the directory
        with os.scandir(directory) as entries:
            for entry in entries:
                # Check if the entry is a directory and if the keyword is in the directory name
                if entry.is_file() and entry.name.endswith(extension):
                    matching_files.append(entry.name)
        
        # Check if there are multiple matching folders
        if len(matching_files) > 1:
            print("Warning: Multiple folders have the extension. Choosing the first one found.")
            for file in matching_files:
                print(f"Matching file: {file}")
        
        # Return the first matching folder or None if no match is found
        if matching_files:
            #print(f"Found and selected file by extension {extension}: {matching_files[0]} ")
            return matching_files[0]
        else:
            print("No file found with extension {extension}.")
            return None

    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
    except PermissionError:
        print(f"Permission denied while accessing {directory}.")
        
        
def create_overwrite_directory(path):
    """Create a directory at the specified path, overwriting if it exists."""
    # Check if the directory exists
    if os.path.exists(path):
        # Remove the directory and all its contents
        shutil.rmtree(path)
        #print(f"Existing directory removed: {path}")

    # Create the new directory
    os.makedirs(path)
    #print(f"New directory created: {path}")

def copy_files_exclude_extensions(src_dir, dest_dir, exclude_extensions):
    """Copy files and directories from src_dir to dest_dir, excluding files with specified extensions."""
    # Create the destination directory if it does not exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Walk through each directory in the source directory
    for dirpath, dirnames, filenames in os.walk(src_dir):
        # Calculate destination path for current directory
        dest_path = os.path.join(dest_dir, os.path.relpath(dirpath, src_dir))
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
            print(f"Created directory {dest_path}")

        # Copy each file within the current directory
        for file in filenames:
            if not file.endswith(tuple(exclude_extensions)):
                src_file_path = os.path.join(dirpath, file)
                dest_file_path = os.path.join(dest_path, file)
                shutil.copy(src_file_path, dest_file_path)
                print(f"Copied {src_file_path} to {dest_file_path}")

        # Optionally, remove the directory names that should not be copied
        # This can be done by modifying dirnames list

def copy_and_rename_tower_file(drive_dir, dummy_tower_filename, template_path):
    """Update the tower file reference in the FAST input file and copy a dummy tower file into the directory."""
    try:
        #-----------------Update fast input-file name--------------------------
        # Find the primary FAST simulation file by extension
        fast_file_name = find_file_by_extension(drive_dir, '.fst')
        fast_path = os.path.join(drive_dir, fast_file_name)
        fst = FASTInputFile(fast_path)  # Load the FAST input file
        
        # Get the ElastoDyn file path from the FAST file
        ed_file = fst['EDFile'][1:-1]  # Removing quotes
        ed_path = os.path.join(drive_dir, ed_file)
        ed = FASTInputFile(ed_path)  # Load the ElastoDyn input object
        
        # Update the tower file entry and write back the changes
        tower_file = ed['TwrFile'][1:-1]  # Removing quotes
        directory, _ = os.path.split(tower_file)
        # Set new tower file path in ElastoDyn object and write to disk
        new_tower_file = os.path.join(directory, dummy_tower_filename)
        ed['TwrFile'] = add_quotationmarks(new_tower_file)
        ed.write()
        
        #-----------------Copy dummy tower to tower directory--------------------------
        # Copy the dummy tower file to the tower directory
        tower_src_path  = os.path.join(template_path, dummy_tower_filename) # Where the dummy-file is located
        tower_dest_path = os.path.join(drive_dir, new_tower_file) # where the elastodyn-file points to
        
        shutil.copy(tower_src_path, tower_dest_path)

        #print(f"Dummy tower file copied to: {tower_dest_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def copy_and_rename_subdyn_file(drive_dir, template_path, subdyn_template_filename):
    """Update the tower file reference in the FAST input file and copy a dummy tower file into the directory."""
    try:
        #-----------------Update fast input-file name--------------------------
        # Find the primary FAST simulation file by extension
        fast_file_name = find_file_by_extension(drive_dir, '.fst')
        subdyn_file_name = fast_file_name.replace('.fst', '_SubDyn.dat')

        fast_path = os.path.join(drive_dir, fast_file_name)
        fst = FASTInputFile(fast_path)  # Load the FAST input file
        fst['SubFile'] = add_quotationmarks(subdyn_file_name)
        fst.write()
        
        #-----------------Copy dummy tower to tower directory--------------------------
        # Copy the dummy tower file to the tower directory
        subdyn_src_path  = os.path.join(template_path, subdyn_template_filename) # Where the dummy-file is located
        subdyn_dest_path = os.path.join(drive_dir, subdyn_file_name) # where the elastodyn-file points to
        shutil.copy(subdyn_src_path, subdyn_dest_path)
        #print(f"SubDyn file template copied to: {subdyn_dest_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    

def update_blade_file_path(drive_dir):
    """Update the blade file reference in the FAST input file if special files for the drivetrain exists."""
    try:
        # Find the primary FAST simulation file by extension
        fast_file_name = find_file_by_extension(drive_dir, '.fst')
        fast_path = os.path.join(drive_dir, fast_file_name)
        fst = FASTInputFile(fast_path)  # Load the FAST input file
        
        # Get the ElastoDyn file path from the FAST file
        ed_file = fst['EDFile'][1:-1]  # Removing quotes
        ed_path = os.path.join(drive_dir, ed_file)
        ed = FASTInputFile(ed_path)  # Load the ElastoDyn input object
        
        # Update the tower file entry and write back the changes
        blade_keys = ['BldFile1', 'BldFile2', 'BldFile3']
        for key in blade_keys:
            blade_file = ed[key][1:-1]
            blade_path = os.path.join(drive_dir, blade_file)
            if os.path.exists(blade_path.replace('.dat', '_drivetrain.dat')):
                new_blade_file = blade_file.replace('.dat', '_drivetrain.dat')
                ed[key] = add_quotationmarks(new_blade_file)
        
        ed.write()
    
    except Exception as e:
        print(f"An error occurred: {e}")


def transform_moi(x1, y1, z1, x2, y2, z2, Ixx, Iyy, Izz):
    """ NB! Assuming local y parallel to global Y"""
    v = np.array([x2 - x1, y2 - y1, z2 - z1])
    v_mag = np.linalg.norm(v)
    v_norm = v / v_mag
    
    local_x = v_norm
    local_y = np.array([0, 1, 0])
    local_z = np.cross(local_x, local_y)

    # Assemble rotation matrix from local to global
    R = np.vstack([local_x, local_y, local_z]).T
    
    I_local_coords = np.array([
        [Ixx, 0, 0], 
        [0, Iyy, 0], 
        [0, 0, Izz]
                              ])

    I_global_coords = R @ (I_local_coords) @ R.T   # CM midlength
    
    return I_global_coords

if __name__ == '__main__':

    # --- INPUT VARIABLES AND CONSTANTS --------------------------------------# 
    tilt = 6 #degs
    x1, y1, z1 = 0, 0, 0
    x2, y2, z2 = -np.cos(tilt), 0, np.sin(tilt)
    transform_moi(x1, y1, z1, x2, y2, z2, 0, 972877, 972877)


