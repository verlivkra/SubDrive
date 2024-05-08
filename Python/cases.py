#! python 3
# -*- coding: utf-8 -*-
"""
Created on 4/9/2024
@author: veronlk

Description of this script.
"""
# --- MODULE IMPORTS ---------------------------------------------------------#
# Example: import numpy as np
import build

# --- FUNCTION DEFINITIONS ---------------------------------------------------#
# def example_function(arg1, arg2):
#    """
#    Example of docstring of a function.
#
#    Arguments:
#        arg1 (float, int):
#            Argument that can be of either float or integer type.
#        arg2 (list of str):
#            Argument that is a list of strings.
#    Returns:
#        return_value (float):
#            Return value of type float.
#    """
#
#    pass

def landbased(script_dir, tower_type, bedplate_type, shaft_type, mb_type, add_nacelle_yaw_inertia = False, add_yaw_br_mass = False):
    mymodel = build.Model(script_dir, platform_type='landbased', interface_location='shaft')
    mymodel.timeStep()
    mymodel.onshoreFixedBase()
    mymodel.buildTower(tower_type = tower_type)
    mymodel.buildBedplate(bedplate_type = bedplate_type)
    mymodel.buildShaft(shaft_type = shaft_type)
    mymodel.buildMainBearing(MB1or2 = 'MB1', mb_type = mb_type)
    mymodel.buildMainBearing(MB1or2 = 'MB2', mb_type = mb_type)
    mymodel.buildGBox()
    mymodel.nacMassIner(add_nacelle_yaw_inertia = add_nacelle_yaw_inertia, add_yaw_br_mass = add_yaw_br_mass) 
    mymodel.outputs()
    mymodel.writeModel()
    mymodel.writeJSON()

def monopile(script_dir, tower_type, bedplate_type, shaft_type, mb_type, add_nacelle_yaw_inertia = False, add_yaw_br_mass = False):
    mymodel = build.Model(script_dir, platform_type='monopile', interface_location='shaft')
    mymodel.timeStep()
    mymodel.monoPile()
    mymodel.buildTower(tower_type = tower_type)
    mymodel.buildBedplate(bedplate_type = bedplate_type)
    mymodel.buildShaft(shaft_type = shaft_type)
    mymodel.buildMainBearing(MB1or2 = 'MB1', mb_type = mb_type)
    mymodel.buildMainBearing(MB1or2 = 'MB2', mb_type = mb_type)
    mymodel.buildGBox()
    mymodel.nacMassIner(add_nacelle_yaw_inertia = add_nacelle_yaw_inertia, add_yaw_br_mass = add_yaw_br_mass) 
    mymodel.outputs()
    mymodel.writeModel()
    mymodel.writeJSON()

def floating(script_dir, tower_type, bedplate_type, shaft_type, mb_type, add_nacelle_yaw_inertia = True, add_yaw_br_mass = True):
    print('Building floating platform')
    mymodel = build.Model(script_dir, platform_type='floating', interface_location='shaft')
    mymodel.timeStep()
    mymodel.floatingPlatform()
    mymodel.buildTower(tower_type = tower_type)
    mymodel.buildBedplate(bedplate_type = bedplate_type)
    mymodel.buildShaft(shaft_type = shaft_type)
    mymodel.buildMainBearing(MB1or2 = 'MB1', mb_type = mb_type)
    mymodel.buildMainBearing(MB1or2 = 'MB2', mb_type = mb_type)
    mymodel.nacMassIner(add_nacelle_yaw_inertia = add_nacelle_yaw_inertia, add_yaw_br_mass = add_yaw_br_mass) 
    mymodel.outputs()
    mymodel.writeModel()
    mymodel.writeJSON()


# --- MAIN SCRIPT STARTS BELOW: ----------------------------------------------#
if __name__ == '__main__':

    # --- INPUT VARIABLES AND CONSTANTS --------------------------------------#

    # --- SCRIPT CONTENT -----------------------------------------------------#
    # IEA 15 MW floating
    floating('tower_from_wisdem', 'wisdem_directdrive')

    # DTU 10MW landbased
    landbased('tower_from_elastodyn', 'straight_beam', add_nacelle_yaw_inertia= False)

    # DTU 10 MW monopile
    monopile('tower_from_elastodyn', 'straight_beam', add_nacelle_yaw_inertia= False)

