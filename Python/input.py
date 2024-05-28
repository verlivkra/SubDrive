#! python 3
# -*- coding: utf-8 -*-
"""
Created on 4/9/2024
@author: veronlk

Description of this script.
"""
# --- MODULE IMPORTS ---------------------------------------------------------#
import numpy as np
import os
import json
import pytest

from openfast_toolbox.io  import FASTInputFile
import tools

# --- FUNCTION DEFINITIONS ---------------------------------------------------#



class FilePaths:
    def __init__(self, 
                 main_path
                 ):
        trad_fol = tools.find_folder_by_keyword(main_path, '_traditional')
        drive_fol = tools.find_folder_by_keyword(main_path, '_drivetrain')
        
        trad_fst = tools.find_file_by_extension(os.path.join(main_path, trad_fol), '.fst')
        drive_fst = tools.find_file_by_extension(os.path.join(main_path, drive_fol), '.fst')

        trad_fst_content = FASTInputFile(os.path.join(main_path, trad_fol, trad_fst))  # Load the FAST input file
        SubFile = trad_fst_content['SubFile']
        sub_path = os.path.join(main_path, trad_fol, SubFile[1:-1]) if SubFile.lower() != "unused" else ""
        self.file_paths = {
            'traditional_fast_path':    os.path.join(main_path, trad_fol, trad_fst),
            'main_fast_path':           os.path.join(main_path, drive_fol, drive_fst),
            'subdyn_monopile_file_path': sub_path,
            'SubFile':                  SubFile,
        }

class BaseFiles:
    """Files from traditional OpenFAST model used as starting points for generation of new OpenFAST model with Subdyn making up the tower and drivetrain"""
    def __init__(self, file_paths):
        
        if not file_paths['subdyn_monopile_file_path'] == '': 
            self.sdMonopileFile = file_paths['subdyn_monopile_file_path']
        
        self.Fst = FASTInputFile(file_paths['traditional_fast_path']) #.fst-object
        self.FstPath = os.path.dirname(file_paths['traditional_fast_path'])

        edFile = self.Fst['EDFile'][1:-1]
        self.Ed = FASTInputFile(os.path.join(self.FstPath, edFile)) #ElastoDyn-object
        self.EdPath ='\\'.join((os.path.join(self.FstPath, edFile)).split('\\')[:-1])
        
        edTwrFile = self.Ed['TwrFile'][1:-1]
        self.edBldFile1 = self.Ed['BldFile1'][1:-1]
        self.edBldFile2 = self.Ed['BldFile2'][1:-1]
        self.edBldFile3 = self.Ed['BldFile3'][1:-1]
        self.EdTw = FASTInputFile(os.path.join(self.EdPath, edTwrFile)) #ElastoDyn Tower-object


class MainFiles:
    """File-name and path to .fst-file making up the OpenFAST model with SubDyn-tower and drivetrain"""
    def __init__(self, file_paths):
        # TODO: These should be automatically named based on basefiles (when running the generator script)
        self.FstPath = file_paths['main_fast_path'] 

class InputParameters():
    """Read user-specified input parameters"""
    def __init__(self, 
                 main_path
                 ):
        input_path = os.path.join(main_path, 'input')
        #General input
        gen_inp = tools.read_yaml(os.path.join(input_path, 'input_parameters.yaml'))
        #Drivetrain input
        drt_file = tools.find_file_by_extension(input_path, '_drivetrain.yaml')

        drt_inp = tools.read_yaml(os.path.join(input_path, drt_file))

        # Time step and simulation duration
        self.TMax   = float(gen_inp['TMax'])
        self.DT_fst = float(gen_inp['DT_fst'])
        self.DT_sd  = float(gen_inp['DT_sd'])
        
        # Input to ElastoDyn
        self.PtfmRIner  = float(gen_inp['PtfmRIner'])
        
        # Damping and modes
        self.DTTorDmp   = float(gen_inp['DTTorDmp'])
        self.CBDamp = list(gen_inp['CBDamp'])
        self.NModes = int(gen_inp['NModes'])
        system_1twr_freq = float(gen_inp['system_1twr_freq']) if gen_inp['system_1twr_freq'] != 'None' else None
        system_1twr_damp = float(gen_inp['system_1twr_damp']) if gen_inp['system_1twr_damp'] != 'None'else None

        if gen_inp['RayleighCoffs'] == 'Calculate':
            beta = SD_beta_from_ED_damp(system_1twr_freq, system_1twr_damp)
            self.RayleighCoffs = [0, beta] 
        else:
            self.RayleighCoffs = gen_inp['RayleighCoffs']
        
        # Tower
        self.TowerE     = float(gen_inp['TowerE']) if gen_inp['TowerE'] != 'None' else None
        self.TowerG     = float(gen_inp['TowerG']) if gen_inp['TowerG'] != 'None' else None
        self.TowerRho   = float(gen_inp['TowerRho']) if gen_inp['TowerRho'] != 'None' else None

        # -------- Drivetrain input -------------
        geom_from_wisdem = bool(drt_inp['Geom_from_WISDEM'])
        if geom_from_wisdem:
            print('Collecting input from wisdem')
            wisdem_inp = tools.read_yaml(os.path.join(input_path, 'wisdem.yaml'))
            self.Lh1    = wisdem_inp['components']['nacelle']['drivetrain']['distance_hub_mb']      # Axial distance from hub flange to MB1
            self.L12    = wisdem_inp['components']['nacelle']['drivetrain']['distance_mb_mb']       # Axial distance from MB1 to MB2
            self.LGen   = wisdem_inp['components']['nacelle']['drivetrain']['generator_length']     # Axial length of generator measured from wall-centers
            self.HhttZ  = wisdem_inp['components']['nacelle']['drivetrain']['distance_tt_hub']      # Hub to tt Z
            self.Dtt    = wisdem_inp['components']['tower']['outer_shape_bem']['outer_diameter']['values'][-1]
            # Calculate common stuff:
            # The following calculations are according to: https://wisdem.readthedocs.io/en/master/wisdem/drivetrainse/layout.html
            # https://github.com/WISDEM/WISDEM/blob/master/wisdem/drivetrainse/layout.py
            self.Lgrs =  self.Lh1/2 # Hub flange to generator rotor side According to WISDEM (reported values don't make sense)
            self.Lgsn = self.LGen - self.Lgrs - self.L12 # Generator stator to bedplate flange According to WISDEM (reported values don't make sense)
            self.L2n = 2*self.Lgsn # MB2 to bedplate flange According to WISDEM (reported values don't make sense)
            self.Llss = self.L12 + self.Lh1 # Length of low speed shaft
            self.Lnose = self.L12 + self.L2n # Length of nose https://github.com/WISDEM/WISDEM/blob/master/wisdem/drivetrainse/layout.py line 259
            self.L_drive = self.Lh1 + self.L12 + self.L2n # Length from bedplate interface to hub interface (Overhang - hubR - bedplate)
            
            # ----------- Material-----------------------------#
            bedplate_material   = wisdem_inp['components']['nacelle']['drivetrain']['bedplate_material']   
            lss_material        = wisdem_inp['components']['nacelle']['drivetrain']['lss_material']        
            for dict in wisdem_inp['materials']:
                if dict['name'] == bedplate_material:
                    self.BdpltE = dict['E']
                    self.BdpltG = dict['G']
                    self.BdpltRho = dict['rho']
                if dict['name'] == lss_material:
                    self.ShftE = dict['E']
                    self.ShftG = dict['G']
                    self.ShftRho = dict['rho']
            
            self.NoseE = self.BdpltE
            self.NoseG = self.BdpltG
            self.NoseRho = self.BdpltRho

            self.ShftOD  = wisdem_inp['components']['nacelle']['drivetrain']['lss_diameter']  
            self.ShftT   = wisdem_inp['components']['nacelle']['drivetrain']['lss_wall_thickness']  
            self.NoseOD  = wisdem_inp['components']['nacelle']['drivetrain']['nose_diameter'] 
            self.NoseT   = wisdem_inp['components']['nacelle']['drivetrain']['nose_wall_thickness'] 
            self.BdpltT = wisdem_inp['components']['nacelle']['drivetrain']['bedplate_wall_thickness']['values']

            # From input-file:
            self.multiplierG    = float(drt_inp['multiplier_G']) # Increase torsional stiffness of drivetrain - should only be captured by 
            self.BdpltG = self.multiplierG*self.BdpltG
            self.ShftG = self.multiplierG*self.ShftG
            self.NoseG = self.multiplierG*self.NoseG

            self.GenStatMass    = float(drt_inp['GenStatMass'])
            self.GenStatMXX     = float(drt_inp['GenStatMXX'])
            self.GenStatMYY     = float(drt_inp['GenStatMYY'])
            self.GenStatMZZ     = float(drt_inp['GenStatMZZ'])
            self.Tow2GenStatX   = float(drt_inp['Tow2GenStatX'])
            self.Tow2GenStatY   = float(drt_inp['Tow2GenStatY'])
            self.Tow2GenStatZ   = float(drt_inp['Tow2GenStatZ'])

            self.GenRotMass    = float(drt_inp['GenRotMass'])
            self.GenRotMXX     = float(drt_inp['GenRotMXX'])
            self.GenRotMYY     = float(drt_inp['GenRotMYY'])
            self.GenRotMZZ     = float(drt_inp['GenRotMZZ'])
            self.Tow2GenRotX   = float(drt_inp['Tow2GenRotX'])
            self.Tow2GenRotY   = float(drt_inp['Tow2GenRotY'])
            self.Tow2GenRotZ   = float(drt_inp['Tow2GenRotZ'])
            
            self.BdpltMassTow2Mid   = float(drt_inp['BdpltMassTow2Mid'])
            self.BdpltMassMid2Nose  = float(drt_inp['BdpltMassMid2Nose'])
            # -------- Main bearing stiffness and cosine matrix must be specified by user even though other drivetrain input comes from wisdem -------------
            self.MB1Spring = {
                'k11': float(drt_inp['Kyy_MB1']), 
                'k22': float(drt_inp['Kzz_MB1']), 
                'k33': float(drt_inp['Kxx_MB1']), 
                'k44': float(drt_inp['Kbb_MB1']), 
                'k55': float(drt_inp['Kgg_MB1']), 
                'k66': float(drt_inp['Kaa_MB1'])
                }
            self.MB2Spring = {
                'k11': float(drt_inp['Kyy_MB2']), 
                'k22': float(drt_inp['Kzz_MB2']), 
                'k33': float(drt_inp['Kxx_MB2']), 
                'k44': float(drt_inp['Kbb_MB2']), 
                'k55': float(drt_inp['Kgg_MB2']), 
                'k66': float(drt_inp['Kaa_MB2'])
                }
            self.MB_cosm        = list(drt_inp['MB_cosm'])
        
        elif drt_file == 'simple_drivetrain.yaml':
            # Shaft 
            ShftProps       = drt_inp['ShaftProps']
            self.ShftE      = float(ShftProps['E'])
            self.ShftG      = float(ShftProps['G'])
            self.ShftRho    = float(ShftProps['rho'])
            self.ShftOD     = float(ShftProps['D_o']) 
            self.ShftT      = float(ShftProps['t'])
            self.ShftL      = float(ShftProps['L']) 
            self.Tow2ShftXGenSide = float(drt_inp['Tow2ShftXGenSide'])

        else:
            # Bedplate
            self.RigidBedplate  = bool(drt_inp['RigidBedplate'])
            self.Bdplt2ShftZ   = float(drt_inp['Bdplt2ShftZ'])
            self.BdpltE         = float(drt_inp['BdpltE']) if drt_inp['BdpltE'] != "unused" else str(drt_inp['BdpltE'])
            self.BdpltG         = float(drt_inp['BdpltG']) if drt_inp['BdpltG'] != "unused" else str(drt_inp['BdpltG'])
            self.BedpltRho      = float(drt_inp['BedpltRho'])
            self.BedpltT        = float(drt_inp['BedpltT'])    # Thickness
            self.BedpltOD       = float(drt_inp['BedpltOD'])   # Outer diameter

            # Generator
            self.Tow2GenX       = float(drt_inp['Tow2GenX'])
            self.Tow2GenY       = float(drt_inp['Tow2GenY'])
            self.Tow2GenZ       = float(drt_inp['Tow2GenZ'])
            self.GenMass        = float(drt_inp['GenMass'])
            self.GenMXX         = float(drt_inp['GenMXX'])
            self.GenMYY         = float(drt_inp['GenMYY'])
            self.GenMZZ         = float(drt_inp['GenMZZ'])
            self.GenMXY         = float(drt_inp['GenMXY'])
            self.GenMXZ         = float(drt_inp['GenMXZ'])
            self.GenMYZ         = float(drt_inp['GenMYZ'])
            
            # Gearbox
            self.Tow2GBMassX    = float(drt_inp['Tow2GBMassX'])
            self.Tow2GBMassY    = float(drt_inp['Tow2GBMassY'])
            self.Tow2GBMassZ    = float(drt_inp['Tow2GBMassZ'])
            self.GBMass         = float(drt_inp['GBMass'])
            self.GBMXX          = float(drt_inp['GBMXX'])
            self.GBMYY          = float(drt_inp['GBMYY'])
            self.GBMZZ          = float(drt_inp['GBMZZ'])
            self.GBMXY          = float(drt_inp['GBMXY'])
            self.GBMXZ          = float(drt_inp['GBMXZ'])
            self.GBMYZ          = float(drt_inp['GBMYZ'])

            # Shaft 
            self.ShftProps      = dict(drt_inp['ShaftProps'])

            # Main bearings
            self.Tow2MB1X       = float(drt_inp['Tow2MB1X'])
            self.Tow2MB2X       = float(drt_inp['Tow2MB2X'])
            
            # GB-Supports
            self.Tow2GBSuppX    = float(drt_inp['Tow2GBSuppX'])
            self.Tow2GBSuppY    = float(drt_inp['Tow2GBSuppY'])
            self.GBSpring = {
                'k11': float(drt_inp['Kyy_GBS']), 
                'k22': float(drt_inp['Kzz_GBS']), 
                'k33': float(drt_inp['Kxx_GBS']), 
                'k44': float(drt_inp['Kbb_GBS']), 
                'k55': float(drt_inp['Kgg_GBS']), 
                'k66': float(drt_inp['Kaa_GBS']), 
            }
            self.GB_cosm        = list(drt_inp['GB_cosm'])

            self.MB1Spring = {
                'k11': float(drt_inp['Kyy_MB1']), 
                'k22': float(drt_inp['Kzz_MB1']), 
                'k33': float(drt_inp['Kxx_MB1']), 
                'k44': float(drt_inp['Kbb_MB1']), 
                'k55': float(drt_inp['Kgg_MB1']), 
                'k66': float(drt_inp['Kaa_MB1'])
                }
            self.MB2Spring = {
                'k11': float(drt_inp['Kyy_MB2']), 
                'k22': float(drt_inp['Kzz_MB2']), 
                'k33': float(drt_inp['Kxx_MB2']), 
                'k44': float(drt_inp['Kbb_MB2']), 
                'k55': float(drt_inp['Kgg_MB2']), 
                'k66': float(drt_inp['Kaa_MB2'])
                }
            self.MB_cosm        = list(drt_inp['MB_cosm'])

        
class Tower:
    def __init__(self, 
                 work_dir, 
                 tower_type = 'tower_from_wisdem'
                 ):
        if tower_type == 'tower_from_wisdem':
            wisdem_path = os.path.join(work_dir, 'input', 'wisdem.yaml')
            self.towerFromWISDEM(wisdem_path)
        elif tower_type == 'tower_from_elastodyn':
            self.towerFromElastoDyn(work_dir)
        elif tower_type == 'tower_uniform':
            self.towerUniform()

    def towerFromWISDEM(self, yaml_path):
        
        """
        Makes SubDyn tower based on yaml-file from WISDEM. 
        NDiv = 1 is used for the entire SubDyn-model.
        If NDiv == 1 and different properties are specified at joint1 and joint2, SubDyn will interpolate the diameter and thickness (?)
        between the joints, but will not allow material properties to change between the two joints
        
        yaml_path (list of dict):
            
        """
        data = tools.read_yaml(file_path = yaml_path)

        #------------ GEOMETRY --------------------#
        tower_geom = data['components']['tower']
        xprops = tower_geom['outer_shape_bem']['reference_axis']['x']
        yprops = tower_geom['outer_shape_bem']['reference_axis']['y']
        zprops = tower_geom['outer_shape_bem']['reference_axis']['z']
        outer_diam = tower_geom['outer_shape_bem']['outer_diameter']
        thickness = tower_geom['internal_structure_2d_fem']['layers'][0]['thickness']

        assert xprops['grid'] == yprops['grid'] == zprops['grid'] == outer_diam['grid'] == thickness['grid']
        self.z = zprops['values']
        self.od = outer_diam['values']
        self.t = thickness['values']
        
        #------------ MATERIAL --------------------#
        # Outfitting factor to multiply with tower material density
        self.outfit_fac = tower_geom['internal_structure_2d_fem']['outfitting_factor']
        
        # Assuming only same material for tower
        self.material_name = tower_geom['internal_structure_2d_fem']['layers'][0]['material']
        
        materials = data['materials']
        for mat in materials:
            if mat['name'] == self.material_name:
                rho = mat['rho']
                self.E = mat['E']
                self.G = mat['G']

        self.rho = rho*self.outfit_fac

    def towerFromElastoDyn(self, work_dir):
        
        """
        Convert elastodyn tower to subdyn tower. Working in the "original" folder to obtain elastodyn data
        NB! Verify resulting tower in SubDyn. This has not been tested substantially. 
        """
        #TODO: This function could use some cleanup and simplifications
        
        # Open file paths and input parameters
        file_manager = FilePaths(work_dir)
        baseF = BaseFiles(file_manager.file_paths)
        inp = InputParameters(work_dir)

        # TODO: Allow for varying material properties along tower
        self.E = inp.TowerE       # Must be constant
        self.G = inp.TowerG        # Must be constant
        self.rho = inp.TowerRho   # Must be constant

        TwHt = baseF.Ed['TowerHt']
        TwBsHt = baseF.Ed['TowerBsHt']
        TowProp = np.transpose(baseF.EdTw['TowProp'])
        HtFract = TowProp[0]
        TMassDen = TowProp[1] #(kg/m)
        TwFAStif = TowProp[2] #(Nm^2) 
        TwSSStif = TowProp[3] #(Nm^2) 
        
        #Check that tower is axisymmetric
        compareStiff = TwFAStif == TwSSStif 

        if compareStiff.all() == False:
            raise Exception("Only axisymmetric members are supported")
        
        self.z   = []
        self.od  = []
        self.t   = []
        for i, htf in enumerate(HtFract):
            self.z.append(htf*(TwHt-TwBsHt) + TwBsHt) 
            D_o = 2*np.sqrt(TMassDen[i]/(2*np.pi*self.rho) + 2*self.rho*TwFAStif[i]/(self.E*TMassDen[i]))
            self.od.append(D_o)
            D_i = 2*np.sqrt((D_o/2)**2-TMassDen[i]/(self.rho*np.pi))
            self.t.append((D_o-D_i)/2)

    def towerUniform(self, TowerProps, NSegments = 10, InterFace = False, TwHt = 100):

        """
        Uniform tower

        """
        
        TwHt = TwHt
        TwBsHt = 0
        
        TwHts = np.arange(0, NSegments+1, 1)*TwHt/(NSegments)
        
        self.BeamProp = []
        self.BPropSetID = 1
        self.BeamProp.append([self.BPropSetID, TowerProps['E'], TowerProps['G'], TowerProps['rho'], TowerProps['D_o'], TowerProps['t']])

        for i, z in enumerate(TwHts):
            skip_joint = False

            # Add node
            if i == 0: #Tower base node is already added to facilitate platform
                skip_joint = True
                MJointID1  = self.twrBsJntID #First joint in new member is twrbsJoint
            else: 
                self.JointID+=1
                self.Joints.append([self.JointID, 0, 0, z, 1, 0.0, 0.0, 0.0, 0.0]) #ID, XSS, YSS, ZSS, JointType (1 = cantilever), JointDirX  JointDirY JointDirZ JointStiff
                MJointID1 = self.JointID-1 #First joint in new member is previous node
                    
            # Add member with jointIDs and PropSetIDs
            if i!=0: #Add new member
                MJointID2 = self.JointID
                if not skip_joint:
                    self.MemberID+=1
                    self.Members.append([self.MemberID, MJointID1, MJointID2, self.BPropSetID, self.BPropSetID, 1, -1]) #MType = 1: Beam
            
            if i == 1: #Tower base member
                self.twrBasMemID = self.MemberID
                
            elif i == len(TwHts)-1: 
                self.twrTopJntID = self.JointID
                self.twrTopMemID = self.MemberID

        #Final joint of the tower is the interface to ElastoDyn.
        if InterFace:
            self.tower_interface = True
            #All fixities must be set to 1 in the current version of SubDyn
            self.InterfaceJoints.append([self.twrTopJntID, 1, 1, 1, 1, 1, 1])

def SD_beta_from_ED_damp(natfreq_system, damp_system): #0.35 for IEA 15 MW (first estimate)
    """ Calculates Rayleigh stiffness proportional damping coefficient, beta, based on system natural frequency (typically 1st FA tower) and ElastoDyn input. """
    beta = np.round(damp_system/(np.pi*natfreq_system), 4)
    return beta

def manualShaftBeamProp():
    """Calculate equivalent shaft material properties and length
    Currently considers Euler-Bernoulli beams only
    Manually input loads and displacements. 
    One beam prop for the entire shaft."""
    
    #-----Euler-Bernoulli with combined moment and force applied in Simpack
    #Combined force and moment Fy and Mz
    Fy_sim = 500000 #N 
    Mz_sim = -5000000 #Nm 
    uy_sim = 0.00227896 #Shaft end horizontal displacement [m]
    rz_sim = -0.00156788 #Shaft end rotation about z-axis [rad]

    Fz_sim = 500000 #N 
    My_sim = 5000000 #Nm 
    uz_sim = 0.00222517 #Shaft end horizontal displacement [m]
    ry_sim = 0.00149648 #Shaft end rotation about z-axis [rad]
    
    Fy_OF = Fz_sim
    Mx_OF = My_sim
    rx_OF = ry_sim
    uy_OF = uz_sim

    # These forces and displacements are used in further calculations - take the average of the two planes
    Fx_OF = np.sign(Fy_sim)*np.mean(np.abs([Fy_sim, Fz_sim]))
    My_OF = np.sign(Mz_sim)*np.mean(np.abs([Mz_sim, My_sim]))
    ry_OF = np.sign(rz_sim)*np.mean(np.abs([rz_sim, ry_sim]))
    ux_OF = np.sign(uy_sim)*np.mean(np.abs([uy_sim, uz_sim]))

    # Used for initial calculation of Leq
    a = -2*Fy_OF*rx_OF 
    b = -3*Mx_OF*rx_OF+3*Fy_OF*uy_OF 
    c = 6*Mx_OF*uy_OF
    Leq1, Leq2 = tools.quadratic_equation(a, b, c)

    a = -2*Fx_OF*ry_OF
    b = 3*My_OF*ry_OF-3*Fx_OF*ux_OF
    c = 6*My_OF*ux_OF
    Leq1, Leq2 = tools.quadratic_equation(a, b, c)

    # Based on the above results, we conveniently choose L as:
    L = 2.82

    Mx_sim = 5000000 #Nm 
    rx_Mx_sim = 0.00119417 #Shaft end rotation about z-axis (beta) [rad] 
    Fx_sim = 500000000 #N 
    ux_Fx_sim = -0.600683-(-0.645001) #Shaft end axial displacement [m] #0.0000001 
     
    Fz_OF = Fx_sim
    uz_Fz_OF = ux_Fx_sim
    
    Mz_OF = Mx_sim
    rz_Mz_OF = rx_Mx_sim
    
    Do2_Di2 = (16*uz_Fz_OF*Fx_OF*L**2)/(Fz_OF*(12*ux_OF+6*L*ry_OF)) #Do**2+Di**2   
    # print("Do2_Di2", Do2_Di2)
    # print("D_i<", np.sqrt(Do2_Di2/2), "<D_o")
    # print("D_o<", np.sqrt(Do2_Di2))
    
    D_o =  1.7 #TODO: Chosen!  #0.0034 
    assert D_o<np.sqrt(Do2_Di2) 
    assert D_o>np.sqrt(Do2_Di2/2)

    D_i = np.sqrt(Do2_Di2-D_o**2)
    # print("D_i<", np.sqrt(Do2_Di2))
    # print("D_i<", np.sqrt(Do2_Di2/2))
    assert D_i<np.sqrt(Do2_Di2/2)
    
    E = (Fz_OF*L)/((np.pi/4)*(D_o**2-D_i**2)*uz_Fz_OF)
    t = (D_o-D_i)/2
    # print("D_o:", D_o, "D_i:", D_i, "E:", E, "t:", t)
    I = (np.pi/64)*(D_o**4-D_i**4)
    # print("I:", I)
    A = (np.pi/4)*(D_o**2-D_i**2)
    G = (231702500000*L)/(2*I)
    # print("G:", G)
    V = A*L
    mass_sim = 30192 #kg
    rho = mass_sim/V
    
    assert 2*G*I/L == pytest.approx(2317025000*100)
    assert E*A*uz_Fz_OF/L == pytest.approx(Fz_OF)
    
    # print(np.abs(12*E*I*ux_OF/(L**3)+6*E*I*ry_OF/(L**2)))
    # print(np.abs(6*E*I*ux_OF/(L**2)+4*E*I*ry_OF/(L)))
    # print(np.abs(12*E*I*uy_OF/(L**3)-6*E*I*rx_OF/(L**2)))
    # print(np.abs(-6*E*I*uy_OF/(L**2)+4*E*I*rx_OF/(L)))

    return {'L': L, 'D_o': D_o, 't': t, 'E': E, 'G': G, 'rho': rho}

def collectShaftBeamProps(JSON_file_path): # TODO: Keep this outside the main script and put in input to input parameters
    """Collect shaft loads and displacements stored in JSON-file. Convert to SubDyn shaft coordinate system"""
    
    #-----Load JSON-------------#
    with open(os.path.join(JSON_file_path)) as file: #, 'displacements.json')) as file:
        file_contents = file.read()
    shaft_props = json.loads(file_contents)
    
    #Combined force and moment Fy and Mz
    Fy_sim = float(shaft_props['Fy Mz']['Fy']) 
    Mz_sim = float(shaft_props['Fy Mz']['Mz']) 
    uy_sim = float(shaft_props['Fy Mz']['ty']) #Shaft end sideways displacement [m]
    rz_sim = float(shaft_props['Fy Mz']['rz']) #Shaft end rotation about z-axis [rad]

    Fz_sim = float(shaft_props['Fz My']['Fz']) 
    My_sim = float(shaft_props['Fz My']['My']) 
    uz_sim = float(shaft_props['Fz My']['tz']) #Shaft end vertical displacement [m]
    ry_sim = float(shaft_props['Fz My']['ry']) #Shaft end rotation about z-axis [rad]

    Fy_sub = -np.sign(Fz_sim)*np.mean(np.abs([Fy_sim, Fz_sim]))
    Mx_sub = -np.sign(My_sim)*np.mean(np.abs([Mz_sim, My_sim]))
    rx_sub = -np.sign(ry_sim)*np.mean(np.abs([rz_sim, ry_sim]))
    uy_sub = -np.sign(uz_sim)*np.mean(np.abs([uy_sim, uz_sim]))

    # These forces and displacements are used in further calculations - take the average of the two planes
    Fx_sub = -np.sign(Fy_sim)*np.mean(np.abs([Fy_sim, Fz_sim]))
    My_sub = -np.sign(Mz_sim)*np.mean(np.abs([Mz_sim, My_sim]))
    ry_sub = -np.sign(rz_sim)*np.mean(np.abs([rz_sim, ry_sim]))
    ux_sub = -np.sign(uy_sim)*np.mean(np.abs([uy_sim, uz_sim]))

    Mx_sim = float(shaft_props['Mx']['Mx']) 
    rx_Mx_sim = float(shaft_props['Mx']['rx'])          #Shaft end rotation about z-axis (beta) [rad] 
    Fx_sim = np.abs(float(shaft_props['Fx']['Fx']))     #N 
    ux_Fx_sim = np.abs(float(shaft_props['Fx']['tx']))  #Shaft end axial displacement [m] 

    Fz_sub = Fx_sim
    uz_Fz_sub = ux_Fx_sim
    Mz_sub = Mx_sim
    rz_Mz_sub = rx_Mx_sim

    forces = [Fx_sub, Fy_sub, Fz_sub]
    moments = [Mx_sub, My_sub, Mz_sub]
    displacements = [ux_sub, uy_sub, uz_Fz_sub]
    rotations = [rx_sub, ry_sub, rz_Mz_sub]

    return forces, moments, displacements, rotations

def calculateShaftBeamPropsEB(JSON_file_path, L_act, D_o_est):
    """Setting the outer diameter is an iterative process, so need to rerun a couple of times to get D_o_est correct. D_o_est is used for the outer diameter in the end. """

    forces, moments, displacements, rotations = collectShaftBeamProps(JSON_file_path)
    
    Fx, Fy, Fz = forces[0], forces[1], forces[2]
    Mx, My, Mz = moments[0], moments[1], moments[2]
    ux, uy, uz = displacements[0], displacements[1], displacements[2]
    rx, ry, rz = rotations[0], rotations[1], rotations[2]

    # Calculate L_equivalent
    a = 2*Fy*rx
    b = 3*(Mx*rx-Fy*uy)
    c = -6*Mx*uy
    Leq1, Leq2 = tools.quadratic_equation(a, b, c)

    L = np.round(Leq1, 3)
    Do2_Di2 = (8*Fx*L**2*uz)/(Fz*(6*ux+3*L*ry)) #Do**2+Di**2   

    Do2_Di2 = (8*My*L*uz)/(Fz*(3*ux+2*L*ry)) #Do**2+Di**2   

    assert Do2_Di2 >= 0
    D_o = D_o_est

    assert D_o<np.sqrt(Do2_Di2) 
    assert D_o>np.sqrt(Do2_Di2/2)

    D_i = np.sqrt(Do2_Di2-D_o**2)
    assert D_i<np.sqrt(Do2_Di2/2)
    
    E = (Fz*L)/((np.pi/4)*(D_o**2-D_i**2)*uz)
    t = (D_o-D_i)/2
    I = (np.pi/64)*(D_o**4-D_i**4)
    A = (np.pi/4)*(D_o**2-D_i**2)
    G = (2317025000*100*L)/(2*I) #TODO: Hard-coded!
    assert E*A*uz/L == pytest.approx(Fz)

    return {'L': L_act, 'D_o': D_o, 't': t, 'E': E, 'G': G}

def calculateShaftBeamRho(ShaftProps):
    V = 0
    for segment, props in ShaftProps.items():
        t = props['t']
        D_o = props['D_o']
        D_i = D_o-2*t
        A = (np.pi/4)*(D_o**2-D_i**2)
        L = props['L']
        V += A*L
    
    mass_sim = 30192 #kg
    rho = mass_sim/V
    for segment in ShaftProps.keys():
        ShaftProps[segment]['rho'] = rho

    return ShaftProps



# --- MAIN SCRIPT STARTS BELOW: ----------------------------------------------#
if __name__ == '__main__':

    # --- INPUT VARIABLES AND CONSTANTS --------------------------------------#

    # --- SCRIPT CONTENT -----------------------------------------------------#

    # Tower(tower_type = 'tower_from_wisdem', wisdem_yaml_path='C:\myGitContributions\OpenFAST-drivetrain-modeling\Python\Tests\IEA15MW_umaine\IEA-15-240-RWT_VolturnUS-S.yaml')
    print('hei')