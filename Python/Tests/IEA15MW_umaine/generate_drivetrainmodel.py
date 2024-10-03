    #! python 3
# -*- coding: utf-8 -*-
"""
Created on 4/9/2024
@author: veronlk

Description of this script.
"""
# --- MODULE IMPORTS ---------------------------------------------------------#
import os
import sys

drtr_module_path = r'C:\Users\veronlk\OneDrive - NTNU\myGitContributions\SubDrive\Python'
if drtr_module_path not in sys.path:
    sys.path.append(drtr_module_path)
import cases, tools



# --- FUNCTION DEFINITIONS ----------# -----------------------------------------#

# --- MAIN SCRIPT STARTS BELOW: ----------------------------------------------#
if __name__ == '__main__':

    # --- INPUT VARIABLES AND CONSTANTS --------------------------------------#
    # predefined_case = 'IEA15MW_floating' # Any of 'DTU10MW_landbased', 'DTU10MW_monopile', 'IEA15MW_floating'
    predefined_case = 'IEA15MW_floating'
    # predefined_case = 'IEA22MW_semi'

    # --- SCRIPT CONTENT -----------------------------------------------------#

    # Move to script location
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    print("Changed working directory to:", os.getcwd())
    print("")

    print("Creating new directory '_drivetrain' and copying files...")
    trad_fol = tools.find_folder_by_keyword(script_dir, '_traditional')
    print("------------------")
    print(trad_fol)
    print("------------------")
    trad_dir = os.path.join(script_dir, trad_fol)

    # Create new directory
    drive_fol = trad_fol.replace('_traditional', '_drivetrain')
    print("------------------")
    print(drive_fol)
    print("------------------")
    drive_dir = os.path.join(script_dir, drive_fol)
    tools.create_overwrite_directory(drive_dir)

    # Specify extensions to exclude (e.g., '.txt', '.jpg')
    extensions_to_exclude = ['.exe', '.sum', '.dbg', '.sum.yaml', '.ech', '.outb', '.out']
    # Copy files, excluding the specified extensions
    tools.copy_files_exclude_extensions(trad_dir, drive_dir, extensions_to_exclude)

    # Add template files 
    common_dir = os.path.dirname(script_dir)
    common_templates_dir = os.path.join(common_dir, 'common_templates')

    # Add dummy tower to tower folder and replace reference in drivetrain elastodyn-model
    dummy_tower_filename = 'ElastoDyn_dummy_tower_light.dat'
    tools.copy_and_rename_tower_file(drive_dir, dummy_tower_filename, common_templates_dir)

    # Update blade file location if necessary
    tools.update_blade_file_path(drive_dir)

    # Add subdyn file
    subdyn_template_filename = 'SubDyn_template_file.dat'
    tools.copy_and_rename_subdyn_file(drive_dir, common_templates_dir, subdyn_template_filename)

    # ------------- Generate drivetrain model ---------------------
    print("Building drivetrain model....")
    if predefined_case == 'DTU10MW_landbased':
        cases.landbased(script_dir, 'tower_from_elastodyn', 'straight_beam', 'DTU10MW_mediumspeed', 'DTU10MW_mediumspeed', add_nacelle_yaw_inertia= False, add_yaw_br_mass = False)
    elif predefined_case == 'DTU10MW_monopile':
        cases.monopile(script_dir, 'tower_from_elastodyn', 'straight_beam', 'DTU10MW_mediumspeed', 'DTU10MW_mediumspeed', add_nacelle_yaw_inertia= False, add_yaw_br_mass = False)
    elif predefined_case == 'IEA15MW_floating_WDtower':
        cases.floating(script_dir, 'tower_from_wisdem', 'wisdem_directdrive', 'wisdem_directdrive', 'wisdem_directdrive', add_nacelle_yaw_inertia= True, add_yaw_br_mass = True)
    elif predefined_case == 'IEA15MW_floating':
        cases.floating(script_dir, 'tower_from_elastodyn', 'wisdem_directdrive', 'wisdem_directdrive', 'wisdem_directdrive', add_nacelle_yaw_inertia= True, add_yaw_br_mass = True)
    elif predefined_case == 'IEA15MW_floating_towerOnly':
        cases.floating_towerOnly(script_dir, 'tower_from_elastodyn', 'wisdem_directdrive', 'wisdem_directdrive', 'wisdem_directdrive', add_nacelle_yaw_inertia= True, add_yaw_br_mass = True)
    elif predefined_case == 'IEA22MW_semi':
        cases.floating(script_dir, 'tower_from_elastodyn', 'wisdem_directdrive', 'wisdem_directdrive', 'wisdem_directdrive', add_nacelle_yaw_inertia= True, add_yaw_br_mass = True)
    elif predefined_case == 'shaft_only':
        cases.monopile_shaft_only(script_dir, 'tower_from_elastodyn', 'none', 'shaft_only', 'shaft_only', add_nacelle_yaw_inertia= True, add_yaw_br_mass = True)

