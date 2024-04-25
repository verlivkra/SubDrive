#! python 3
# -*- coding: utf-8 -*-
"""
Created on 4/9/2024
@author: veronlk

Script that builds the ElastoDyn and SubDyn drivetrain/turbine/platform model. 

"""
# --- MODULE IMPORTS ---------------------------------------------------------#
import numpy as np
import tools
from openfast_toolbox.io  import FASTInputFile
import os
import json
import sys
import input
import pytest
import scipy.integrate as integrate

# --- FUNCTION DEFINITIONS ---------------------------------------------------#

class Model:
    def __init__(self, work_dir, platform_type = 'floating', 
                 interface_location = 'shaft', 
                 ): 
        """
        Class that builds the combined ElastoDyn and SubDyn drivetrain/turbine model
        
        Arguments:
            work_dir (str):
                path to folder where input folder and template folders exist
            platform_type (str):
                Can be either "landbased", "monopile" or "floating"
            interface_location (str):
                Location of the interface joint.
                Can be either 
                    "shaft" (shaft tip on rotor side), 
                    "bedplate" (bedplate rotor side end) or 
                    "tower" (tower top). 
                Note that "bedplate" and "tower" is only used for testing. 
        """

        # ID sets
        self.BPropSetID         = 0 # Beam propset ID
        self.JointID            = 0 
        self.MemberID           = 0
        self.CosmID             = 0 # Cosine matrix ID
        self.RPropSetID         = 1 # Rigid link propset ID
        self.SPropSetID         = 0 # Spring propset ID

        self.ConcentratedMasses     = []
        self.InterfaceJoints        = []
        self.BaseReactionJoints     = []
        self.Members                = []
        self.Joints                 = []
        self.Cosms                  = []
        self.BeamProp               = []
        self.SpringProp             = []

        # Create massless rigid property
        self.RigidProp              = [[self.RPropSetID, 0]] #Zero material density
        self.RPropSetIDMassless = 1
        
        # Main bearing cosine matrix
        self.MBCosmID = False # Before created
        
        # Keep track of drivetrain accumulated mass, center of mass and inertia to get correct nacelle mass in ElastoDyn and SubDyn: 
        self.drTrMassProps       = []  #CMx, CMy, CMz, mass, Ixx, Iyy, Izz

        # Initiate short stiff beam
        self.stiffBeamProp()
        
        # List of gearbox support memberIDs
        self.GBSuppMemID = []
        
        # Floating, monopile or landbased platform
        self.platformType = platform_type
        print(self.platformType)
        self.interfaceLocation = interface_location

        #----------- File handling -------------------
        self.work_dir = work_dir
        file_manager = input.FilePaths(self.work_dir)
        # Reading user-defined input parameters
        self.inputs = input.InputParameters(self.work_dir)
        # Traditional OpenFAST-files - used as template and provides general input about the turbine
        self.baseF = input.BaseFiles(file_manager.file_paths)
        # Path to model files
        self.mainF = input.MainFiles(file_manager.file_paths)

        # Initialize/open OpenFAST modules
        self.fstModules()

    def fstModules(self):
        """Initialize FAST input files"""

        self.fstName            = os.path.basename(self.mainF.FstPath)
        self.mainPath           = os.path.dirname(self.mainF.FstPath) # path to directory of OpenFAST model with Subdyn
        self.fst                = FASTInputFile(self.mainF.FstPath) #.fst-object

        edFile = self.fst['EDFile'][1:-1]
        self.edPath = os.path.join(self.mainPath, edFile)
        self.ed  = FASTInputFile(self.edPath)
        
        sdFile = self.fst['SubFile'][1:-1]
        self.sdPath = os.path.join(self.mainPath, sdFile)
        self.sd = FASTInputFile(self.sdPath)
            
        if self.fst['HydroFile'].lower() != '"unused"': 
            hdFile = self.fst['HydroFile'][1:-1]
            self.hdPath = os.path.join(self.mainPath, hdFile)
            self.hd = FASTInputFile(self.hdPath)
            
        if self.fst['MooringFile'].lower() != '"unused"':  #Used to find connection points
            mdFile = self.fst['MooringFile'][1:-1]
            self.mdPath = os.path.join(self.mainPath, mdFile)
            self.md = FASTInputFile(self.mdPath)

        if self.fst['InflowFile'].lower() != '"unused"':            
            iwFile = self.fst['InflowFile'][1:-1]
            self.iwPath = os.path.join(self.mainPath, iwFile)
            self.iw = FASTInputFile(self.iwPath)

    def stiffBeamProp(self): #TODO: lower-case for all functions, separate by _
        """Function that creates a generic stiff beam"""

        E = 210000000000000
        G = 8080000000000
        rho = 1 #TODO: Can we reduce this?
        D_o = 1
        t = 0.4
        self.BPropSetID+=1
        self.BeamProp.append([self.BPropSetID, E, G, rho, D_o, t])
        self.StiffBeamPropID = self.BPropSetID

    def timeStep(self):
        """Function that modifies timestep of fst-file, SubDyn-file, ElastoDyn-file and HydroDyn-file"""
        
        self.fst['TMax'] = self.inputs.TMax
        self.fst['DT'] = self.inputs.DT_fst
        self.ed['DT'] = self.inputs.DT_fst
        self.sd['SDdeltaT'] = self.inputs.DT_sd  
        
        if hasattr(self, 'hd'):
            self.hd['RdtnDT'] = self.inputs.DT_fst
            #self.hd['WaveTMax'] = self.inputs.TMax

    def towerBaseJoint(self):
        """Joint connecting tower and mooring lines. At tower base."""

        TowerBsHt = self.baseF.Ed['TowerBsHt']
        self.JointID+=1
        self.Joints.append([self.JointID, 0, 0, TowerBsHt, 1, 0.0, 0.0, 0.0, 0.0]) 
        self.twrBsJntID=self.JointID

    def assertions(self):
        """Various tests and assertions"""
        assert tools.unique_val_list(np.transpose(self.Joints)[0])
        assert tools.unique_val_list(np.transpose(self.Members)[0])
        assert tools.unique_val_list(np.transpose(self.BeamProp)[0])
        assert tools.unique_val_list(np.transpose(self.RigidProp)[0])
        assert tools.unique_val_list(np.transpose(self.ConcentratedMasses)[0])
        #TODO: Add check to see that all nodes are connected to a member

    def floatingPlatform(self):
        """
        Add rigid elements between fairleads and tower in new Subdyn file, based on original ElastoDyn file
        """
        fairLeadJts = []
        #Add fairlead joints
        for con in self.md['Points']:
            if con[1] == 'Vessel':
                self.JointID += 1
                self.Joints.append([self.JointID, float(con[2]), float(con[3]), float(con[4]), 1, 0.0, 0.0, 0.0, 0.0]) #ID, XSS, YSS, ZSS, JointType (1 = cantilever), JointDirX  JointDirY JointDirZ JointStiff            
                fairLeadJts.append(self.JointID) #TODO add these to self? #Fairlead joints
                
        #Node for platform mass and inertia. Found from "original" ED-files
        self.JointID += 1
        self.Joints.append([self.JointID, self.baseF.Ed['PtfmCMxt'], self.baseF.Ed['PtfmCMyt'], self.baseF.Ed['PtfmCMzt'], 1, 0.0, 0.0, 0.0, 0.0]) #ID, XSS, YSS, ZSS, JointType (1 = cantilever), JointDirX  JointDirY JointDirZ JointStiff            
        # Mass and inertia applied relative to the SS coordinate system (global inertial-frame coordinate system) [kg], [kgm^2]
        self.ConcentratedMasses.append([self.JointID, self.baseF.Ed['PtfmMass'], self.baseF.Ed['PtfmRIner'], self.baseF.Ed['PtfmPIner'], 
                                        self.baseF.Ed['PtfmYIner'], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ptfmCMJtID = self.JointID 
        self.towerBaseJoint() #Establish position of tower base joint and add joint
        
        #Connect fairleads to tower base
        for fJtID in fairLeadJts:
            self.MemberID +=1 
            self.Members.append([self.MemberID, fJtID, self.twrBsJntID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 3: Rigid link
       
        #Connect ptfm center of mass to tower base
        self.MemberID +=1 
        self.Members.append([self.MemberID, self.ptfmCMJtID, self.twrBsJntID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 3: Rigid link

    def monoPile(self):
        """
        Create a monopile based on a subdyn template-file, including beam elements and reaction joint.
        Create tower base joint.
        """
        sd = FASTInputFile(self.baseF.sdMonopileFile)
        #----- Joints -----#
        reactionJt = sd['BaseJoints'][0] # NB! Assuming only 1 BaseJoint

        z_min = 100000
        for i, jt in enumerate(sd['Joints']):
            self.JointID += 1
            self.Joints.append([self.JointID, jt[1], jt[2], jt[3], jt[4], jt[5], jt[6], jt[6], jt[7]])
            if jt[3] < z_min:
                z_min = jt[3]
                self.pileBsJntID=self.JointID # Monopile bottom joint
            if i == len(sd['Joints'])-1: # Joint linking monopile to tower base
                self.twrBsJntID=self.JointID
            elif i == 0:
                self.BaseReactionJoints.append([self.JointID,    reactionJt[1],      reactionJt[2],      reactionJt[3],      reactionJt[4],      reactionJt[5],      reactionJt[6],      reactionJt[7]])
        
        #----- Beam properties -----#
        propIDOld2New = {} # Keep track of beamprop IDs in template file to make sure members get updated beam properties
        for prop in sd['BeamProp']:
            self.BPropSetID += 1
            self.BeamProp.append([self.BPropSetID, prop[1], prop[2], prop[3], prop[4], prop[5]]) #MType = 3: Rigid link
            propIDOld2New[prop[0]] = self.BPropSetID

        #----- Members -----#
        for mem in sd['Members']:
            self.MemberID += 1
            if mem[1] == self.pileBsJntID:
                self.pileBsMemID = self.MemberID
            self.Members.append([self.MemberID, mem[1], mem[2], propIDOld2New[mem[3]], propIDOld2New[mem[4]], mem[5], -1]) #MType = 3: Rigid link
        
        return
    
    def onshoreFixedBase(self):
        """
        Add base reaction joint at tower base
        """
        self.towerBaseJoint() #Establish position of tower base joint and add joint
        
        #Connect ptfm center of mass to tower base
        self.BaseReactionJoints.append([self.twrBsJntID,    1,      1,      1,      1,      1,      1,      '""'])

    def jointByID(self, ID):
        """Get joint properties based on ID"""
        for joint in self.Joints:
            if joint[0] == ID:
                break
        return joint
    
    def propByID(self, ID):
        """Get beam properties based on ID"""
        for prop in self.BeamProp:
            if prop[0] == ID:
                break
        return prop
    
    def membByID(self, ID):
        """Get member properties based on ID"""
        for member in self.Members:
            if member[0] == ID:
                break
        return member

    def pipeInpByMemID(self, memberID, TTZ = 0):
        [_, MJointID1, MJointID2, BPropSetID1, BPropSetID2, _, _] = self.membByID(memberID)
        
        jntCrds1 = self.jointByID(MJointID1)[1:4]
        jntCrds2 = self.jointByID(MJointID2)[1:4]

        [_, _, _, rho1, OD1, t1] = self.propByID(BPropSetID1)
        [_, _, _, rho2, OD2, t2] = self.propByID(BPropSetID2)
        rho = rho1 # Assuming uniform mass density

        pipe = tools.TaperedPipe(rho, OD1, OD2, t1, t2, jntCrds1, jntCrds2, True, TTZ)
        return pipe

    # def calcBeamMassProps(self, MemberID):
        
    #     memberProps = self.membByID(MemberID)
    #     jntID1 = memberProps[1]
    #     jntID2 = memberProps[2]
    #     PropID1 = memberProps[3]
    #     PropID2 = memberProps[4]
    #     memType = memberProps[5]
    #     if memType != 1:
    #         print('WARNING! calcBeamMassProps should only be used for beams!')
    #         print('The warning relates to member {}'.format(MemberID))
        
    #     BeamProps1 = self.propByID(PropID1) 
    #     BeamProps2 = self.propByID(PropID2) 
        
    #     if BeamProps1[3] != BeamProps2[3]:
    #         print("WARNING! The function 'calcBeamMassProps' assumes continues material properties in a member. If this is not the case, it will use the beam properties of joint 1")
        
    #     #Member length
    #     jnt1        = self.jointByID(jntID1)
    #     x1, y1, z1  = jnt1[1], jnt1[2], jnt1[3]
    #     print(x1, y1, z1)
    #     jnt2        = self.jointByID(jntID2)
    #     x2, y2, z2  = jnt2[1], jnt2[2], jnt2[3]
    #     print(x2, y2, z2)
    #     L           = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
        
    #     #Member density
    #     MatDens = BeamProps1[3]

    #     # Mass of (possibly) tapered beam
    #     A1 = tools.pipe_area(BeamProps1[4], BeamProps1[5])
    #     print("OD", "t")
    #     print(BeamProps1[4], BeamProps1[5])
    #     A2 = tools.pipe_area(BeamProps2[4], BeamProps2[5])
    #     print("OD", "t")
    #     print(BeamProps2[4], BeamProps2[5])
        
    #     # TODO: Verify by numbers!
    #     V = L/3*(A1 + np.sqrt(A1*A2) + A2)
    #     mass    = V*MatDens
        
    #     # Normalized orientation vector
    #     v = np.array([x2 - x1, y2 - y1, z2 - z1])
    #     v_mag = np.linalg.norm(v)
    #     v_norm = v / v_mag

    #     # Center of mass
    #     # TODO: Verify by numbers!
    #     CMz_local = (A1 * L**2 / 2 + L**2 * (A2 - A1) / 3) / (A1 * L + L * (A2 - A1) / 2) # Along longitudinal axis of member
    #     print("CMx_local", CMz_local)                
    #     CMx = x1 + CMz_local * v_norm[0]
    #     CMy = y1 + CMz_local * v_norm[1]
    #     CMz = z1 + CMz_local * v_norm[2]

    #     # Diameter as a function of x
    #     def diameter(x, D1, D2, L):
    #         return D1 + (D2 - D1) * (x / L)

    #     # Local moment of inertia functions
    #     def I_zz(x):
    #         d = diameter(x)
    #         return (np.pi * d**4) / 64

    #     def I_xx_yy(x):
    #         d = diameter(x)
    #         return (np.pi * d**4) / 32

    #     # Density function
    #     def density(x, rho):
    #         return rho * (np.pi * (diameter(x, d1, d2, l) / 2)**2)

    #     # Composite functions for integration
    #     def integrand_zz(x, rho):
    #         return density(x, rho) * I_zz(x)

    #     def integrand_xx_yy(x):
    #         return density(x, rho) * I_xx_yy(x)
        
    #     # Integrate to find local moments
    #     I_local_zz = integrate.quad(integrand_zz, 0, L)[0]
    #     I_local_xx_yy = integrate.quad(integrand_xx_yy, 0, L)[0]

    #     # Calculate mass moment of inertia in global coordinate system
    #     local_z = v_norm # local z-axis along length of beam
    #     # Construct orthogonal local x and y axes using Gram-Schmidt process (simplifying assumptions)
    #     local_x = np.array([-local_z[1], local_z[0], 0])
    #     local_x /= np.linalg.norm(local_x)
    #     local_y = np.cross(local_z, local_x)

    #     # Assemble rotation matrix from local to global
    #     R = np.vstack([local_x, local_y, local_z]).T

    #     # Apply rotation matrix to transform local inertia
    #     I_global = R @ np.diag([I_local_xx_yy, I_local_xx_yy, I_local_zz]) @ R.T
        
    #     print(f"Coordinates of member {MemberID}: X1, X2 = {x1, x2}, Y1, Y2 = {y1, y2}, Z1, Z2 = {z1, z2}")
    #     print(f"Global Center of Mass Coordinates of member {MemberID}: X_cm = {CMx}, Y_cm = {CMy}, Z_cm = {CMz}")

    #     return [CMx, CMy, CMz, mass]
    
    def buildTower(self, tower_type = 'tower_from_wisdem'):
        tower = input.Tower(self.work_dir, tower_type = tower_type)

        # Assert tower height
        TwHt = self.baseF.Ed['TowerHt']
        TwBsHt = self.baseF.Ed['TowerBsHt']
        assert TwHt == tower.z[-1]
        assert TwBsHt == tower.z[0]

        # Trackers
        prev_D_o = 0
        prev_t = 0
        z_0 = 0

        for i, z in enumerate(tower.z):
            skip_joint = False
            prev_flag = 0
            # Add joint
            if i == 0: #Tower base node is already added to facilitate platform #TODO and z == TwBsHt
                skip_joint = True
                MJointID1  = self.twrBsJntID #First joint in new member is twrbsJoint
            else: 
                if z-z_0<0.1: #Very short segment. Skip
                    skip_joint = True
                else: 
                    self.JointID+=1
                    self.Joints.append([self.JointID, 0, 0, z, 1, 0.0, 0.0, 0.0, 0.0]) #ID, XSS, YSS, ZSS, JointType (1 = cantilever), JointDirX  JointDirY JointDirZ JointStiff
                MJointID1 = self.JointID-1 #First joint in new member is previous node
            # Add unique beam property set to property sets
            D_o = tower.od[i]
            t = tower.t[i]
            
            # TODO: Currently only checking that this property set is not a duplicate of the previous set. 
            if D_o == prev_D_o and t == prev_t:
                prev_flag = 1 #Used for propIDs for the members
            else: 
                self.BPropSetID += 1
                self.BeamProp.append([self.BPropSetID, tower.E, tower.G, tower.rho, np.round(D_o, 4), np.round(t, 4)])
            
            #Cheap way of keeping track of previous property set to avoid duplicate property sets 
            prev_D_o = D_o
            prev_t = t
            
            # Add members with jointIDs and PropSetIDs
            if i!=0: #Add new member
                MJointID2 = self.JointID
                if skip_joint:
                    pass
                else: 
                    if prev_flag: # Same propsetID for both joints
                        self.MemberID+=1
                        self.Members.append([self.MemberID, MJointID1, MJointID2, self.BPropSetID, self.BPropSetID, 1, -1]) #MType = 1: Beam
                    else: 
                        self.MemberID+=1
                        self.Members.append([self.MemberID, MJointID1, MJointID2, self.BPropSetID-1, self.BPropSetID, 1, -1]) #MType = 1: Beam
            
            if i == 1: #Tower base member
                self.twrBasMemID = self.MemberID
                
            #Final joint of the tower is the interface to ElastoDyn. 
            if i == len(tower.z)-1: 
                self.twrTopJntID=self.JointID
                self.twrTopMemID = self.MemberID

            z_0 = z
        if self.interfaceLocation == 'tower':
            #All fixities must be set to 1 in the current version of SubDyn
            self.InterfaceJoints.append([self.twrTopJntID, 1, 1, 1, 1, 1, 1])

    def buildBedplate(self, bedplate_type = 'straight_beam'):
        """Build bedplate in SubDyn"""

        if bedplate_type == 'straight_beam':
            self.bedplate_SB()
        elif bedplate_type == 'wisdem_directdrive':
            self.bedplate_DD_wisdem()


    def bedplate_SB(self): #TODO: Could use some clean-up
        """
        This bedplate is a straight, horizontal beam. 
        Currently only a rigid bedplate is supported. 
        This means that only its mass, and joints to bearings, gearboxes and generators are important.
        """
        RigidBedplate = self.inputs.RigidBedplate

        if RigidBedplate == True and self.interfaceLocation == 'bedplate':
            print('Warning: A rigid bedplate will not work together with having the interface joint in the bedplate. Rigid links doesnt work together with InterFace joints.')
            print('Setting interface_location to "tower".')
            self.interfaceLocation = 'tower'

        #Input and geometry
        TwrTopJoint = self.jointByID(self.twrTopJntID)
        TwrTopX = TwrTopJoint[1]
        TwrTopZ = TwrTopJoint[3]
        
        Tow2BedpltZ = self.baseF.Ed['Twr2Shft']-self.inputs.Bdplt2ShftZ
        BedPltZ = TwrTopZ+Tow2BedpltZ

        #Generator inertia about shaft is included in ElastoDyn. 
        #Here, JMXX is inertia in global coord due to the shaft not being aligned with the inertial reference frame (as calculated in Simpack w/o Ixx in local coord. system)
        BedpltJoints = {
            'MB1': {'xyz': [self.inputs.Tow2MB1X, 0, "unused"]},
            'MB2': {'xyz': [self.inputs.Tow2MB2X, 0, "unused"]},
            'GB':  {'xyz': [self.inputs.Tow2GBSuppX, 0, "unused"]},
            'Gen': {'xyz': [self.inputs.Tow2GenX, self.inputs.Tow2GenY, self.inputs.Tow2GenZ + self.baseF.Ed['TowerHt']], 
                    'Mass': {'JMass': self.inputs.GenMass, 
                             'JMXX': self.inputs.GenMXX, 'JMYY': self.inputs.GenMYY, 'JMZZ': self.inputs.GenMZZ, 
                             'JMXY': self.inputs.GenMXY, 'JMXZ': self.inputs.GenMXZ, 'JMYZ': self.inputs.GenMYZ, 
                             'MCGX': 0, 'MCGY': 0, 'MCGZ': 0}} #Inertias calculated using Ixx = 0 for generator
            }
        BedpltLtot = BedpltJoints['Gen']['xyz'][0]-self.inputs.Tow2MB1X
        
        #Rigid link between tower top and bedplate
        self.JointID += 1
        self.Joints.append([self.JointID, TwrTopX, 0, BedPltZ, 1, 0.0, 0.0, 0.0, 0.0])
        self.bplt2TwrJntID = self.JointID
        self.MemberID+=1
        
        if self.interfaceLocation == 'tower':  #Cannot have rigid link connected to tower top joint if interface is to be at tower top too. Use stiff beam instead. Typically only for debugging.
            self.Members.append([self.MemberID, self.bplt2TwrJntID, self.twrTopJntID, self.StiffBeamPropID , self.StiffBeamPropID , 1, -1]) #MType = 1: Beam
        else: 
            self.Members.append([self.MemberID, self.bplt2TwrJntID, self.twrTopJntID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 1: Beam
        
        #Bedplate material properties
        #Currently only one material property set for entire bedplate
        if RigidBedplate:
            self.RPropSetID+=1
            rigPropA = np.pi/4*(self.inputs.BedpltOD**2-(self.inputs.BedpltOD-2*self.inputs.BedpltT)**2)
            self.RigidProp.append([self.RPropSetID, self.inputs.BedpltRho*rigPropA])
            BP_PropSetID = self.RPropSetID
            BP_MType = 3
        else:
            print('Warning: Currently only a rigid bedplate is supported for this drivetrain. Bedplate will be rigid.')
            RigidBedplate = True
            # self.BPropSetID+=1
            # self.BeamProp.append([self.BPropSetID, self.BedpltE, self.BedpltG, self.BedpltRho, self.BedpltOD, self.BedpltT])
            # BP_PropSetID = self.BPropSetID
            # BP_MType = 1

        # Joint for each intersection with e.g. gearbox, generator etc. 
        for i, jntName in enumerate(BedpltJoints.keys()): 
            
            jnt = BedpltJoints[jntName]

            jntCrds = jnt['xyz']
            jntX, jntY, jntZ = jntCrds[0], jntCrds[1], jntCrds[2]
            if jntName == 'MB1':
                MB1_x = jntX
                MB1_y = 0
                MB1_z = BedPltZ
            elif jntName == 'Gen': 
                Gen_x = jntX
                Gen_y = 0
                Gen_z = BedPltZ
            #Upstream of tower
            if jntX<TwrTopX:
                if jntZ != "unused": # Joint is positioned above or below bedplate -> add rigid link to joint
                    
                    # Add joint off bedplate first
                    self.JointID += 1
                    self.Joints.append([self.JointID, jntX, jntY, jntZ, 1, 0.0, 0.0, 0.0, 0.0])
                    if 'Mass' in jnt.keys():
                        mass = jnt['Mass']
                        self.ConcentratedMasses.append([self.JointID, mass['JMass'], mass['JMXX'], mass['JMYY'], mass['JMZZ'], 
                                                   mass['JMXY'], mass['JMXZ'], mass['JMYZ'], 
                                                   mass['MCGX'], mass['MCGY'], mass['MCGZ']])
                        self.drTrMassProps.append([jntX, jntY, jntZ, mass['JMass'], mass['JMXX'], mass['JMYY'], mass['JMZZ']]) 
                    self.JointID+=1
                    self.Joints.append([self.JointID, jntX, 0, BedPltZ, 1, 0.0, 0.0, 0.0, 0.0]) #Assume bedplate is along y = 0

                    self.MemberID+=1
                    # if RigidLinks:
                    self.Members.append([self.MemberID, self.JointID-1, self.JointID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 3: Rigid link
                    # else:
                    #     self.Members.append([self.MemberID, self.JointID-1, self.JointID, self.StiffBeamPropID , self.StiffBeamPropID , 1, -1]) #MType = 1: Beam
                    
                    if jntName != 'MB1': 
                        self.MemberID+=1
                        self.Members.append([self.MemberID, self.JointID-2, self.JointID, BP_PropSetID, BP_PropSetID, BP_MType, -1]) #MType = 3: Rigid link
                else: 
                    
                    self.JointID+=1
                    self.Joints.append([self.JointID, jntX, 0, BedPltZ, 1, 0.0, 0.0, 0.0, 0.0]) 

                    if i != 0:
                        self.MemberID+=1
                        self.Members.append([self.MemberID, self.JointID-1, self.JointID, BP_PropSetID, BP_PropSetID, BP_MType, -1]) #MType = 1: Beam
                    if jntName == 'MB1': #MB1 should not be connected to previous node, which is the tower top!
                        self.MB1BedpltJtID = self.JointID #Connection nodes for the bearings
                    if jntName == 'MB2':
                        self.MB2BedpltJtID = self.JointID #Connection nodes for the bearings
                    if jntName == 'GB':
                        self.GBBdpltJtID = self.JointID #Connection nodes for the bearings

            if i == 0 and self.interfaceLocation == 'bedplate': #Not valid for rigid bedplate
                self.InterfaceJoints.append([self.JointID, 1, 1, 1, 1, 1, 1])  #All fixities must be set to 1 in the current version of SubDyn

        #Link tower top and last upstream coordinate in bedplate:
        self.MemberID+=1
        self.Members.append([self.MemberID, self.JointID, self.bplt2TwrJntID, BP_PropSetID, BP_PropSetID, BP_MType, -1]) #MType = 1: Beam
        frst_dwnstrm = True

        for i, jntName in enumerate(BedpltJoints.keys()): 
            
            jnt = BedpltJoints[jntName]

            jntCrds = jnt['xyz']
            jntX, jntY, jntZ = jntCrds[0], jntCrds[1], jntCrds[2]
            
            if jntName == 'MB1':
                MB1_x = jntX
                MB1_y = 0
                MB1_z = BedPltZ
            elif jntName == 'Gen': 
                Gen_x = jntX
                Gen_y = 0
                Gen_z = BedPltZ
            
            #Downstream of tower
            if jntX>TwrTopX:
                if jntZ != "unused": #z != tower top -> add rigid link! 
                    
                    #Add joint off bedplate
                    self.JointID+=1
                    self.Joints.append([self.JointID, jntX, jntY, jntZ, 1, 0.0, 0.0, 0.0, 0.0])
                    if 'Mass' in jnt.keys():
                        mass = jnt['Mass']
                        self.ConcentratedMasses.append([self.JointID, mass['JMass'], mass['JMXX'], mass['JMYY'], mass['JMZZ'], 
                                                   mass['JMXY'], mass['JMXZ'], mass['JMYZ'], 
                                                   mass['MCGX'], mass['MCGY'], mass['MCGZ']])
                        self.drTrMassProps.append([jntX, jntY, jntZ, mass['JMass'], mass['JMXX'], mass['JMYY'], mass['JMZZ']]) 
                    
                    self.JointID +=1
                    self.Joints.append([self.JointID, jntX, 0, BedPltZ, 1, 0.0, 0.0, 0.0, 0.0]) #Assume bedplate is along y = 0
                    
                    self.MemberID +=1
                    
                    # if RigidLinks:
                    self.Members.append([self.MemberID, self.JointID-1, self.JointID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 3: Rigid link
                    # else:
                        # self.Members.append([self.MemberID, self.JointID-1, self.JointID, self.StiffBeamPropID , self.StiffBeamPropID , 1, -1]) #MType = 1: Beam
                    
                    if frst_dwnstrm:
                        self.MemberID+=1
                        self.Members.append([self.MemberID, self.bplt2TwrJntID, self.JointID, BP_PropSetID, BP_PropSetID, BP_MType, -1]) #MType = 1: Beam
                        frst_dwnstrm = False
                    else: 
                        self.MemberID+=1
                        self.Members.append([self.MemberID, self.JointID-2, self.JointID, BP_PropSetID, BP_PropSetID, BP_MType, -1]) #MType = 3: Rigid link
                else: 
                    self.JointID+=1
                    self.Joints.append([self.JointID, jntX, 0, BedPltZ, 1, 0.0, 0.0, 0.0, 0.0]) 
                    
                    if jntName == 'MB1':
                        self.MB1_bedplate = self.JointID #Connection nodes for the bearings
                    elif jntName == 'MB2':
                        self.MB2_bedplate = self.JointID #Connection nodes for the bearings
                    elif jntName == 'GB':
                        self.GBBdpltJtID = self.JointID #Connection nodes for the bearings
                    if frst_dwnstrm: 
                        #First coordinate downstream of tower:
                        self.MemberID+=1
                        self.Members.append([self.MemberID, self.bplt2TwrJntID, self.JointID, BP_PropSetID, BP_PropSetID, BP_MType, -1]) #MType = 1: Beam
                        frst_dwnstrm = False
                    else: 
                        self.MemberID+=1
                        self.Members.append([self.MemberID, self.JointID-1, self.JointID, BP_PropSetID, BP_PropSetID, BP_MType, -1]) #MType = 1: Beam
    
        # Sum up bedplate mass        
        bdpltLength = BedpltLtot
        bdpltXArea  = (np.pi/4)*(self.inputs.BedpltOD**2-(self.inputs.BedpltOD-2*self.inputs.BedpltT)**2)
        bdpltMass = self.inputs.BedpltRho*bdpltLength*bdpltXArea
        # Uniform mass, assume similar beamprops along the whole shaft
        bdpltCMx = Gen_x-(Gen_x-MB1_x)/2
        bdpltCMy = Gen_y-(Gen_y-MB1_y)/2
        bdpltCMz = Gen_z-(Gen_z-MB1_z)/2
        
        R_i = (self.inputs.BedpltOD-2*self.inputs.BedpltT)/2
        R_o = self.inputs.BedpltOD/2
        
        #Inertia about bedplate COG in global coordinate system
        bdpltIxx = 1/2*bdpltMass*(R_i**2+R_o**2)
        bdpltIyy = 1/12*bdpltMass*(3*(R_i**2+R_o**2) + bdpltLength**2)  # TODO: Is this correct according to Steiner?
        bdpltIzz = 1/12*bdpltMass*(3*(R_i**2+R_o**2) + bdpltLength**2)  
        
        self.drTrMassProps.append([bdpltCMx, bdpltCMy, bdpltCMz, bdpltMass, bdpltIxx, bdpltIyy, bdpltIzz]) 



    def bedplate_DD_wisdem(self): #TODO: Could use some clean-up
        # TODO: Checkout: https://wisdem.readthedocs.io/en/master/wisdem/drivetrainse/components.html#bedplate
        """
        This bedplate is a straight, horizontal beam. 
        Currently only a rigid bedplate is supported. 
        This means that only its mass, and joints to bearings, gearboxes and generators are important.
        """

        #Input and geometry
        ShftTilt    = -self.baseF.Ed['ShftTilt']
        TowerHt     = self.baseF.Ed['TowerHt']
        HubRad      = self.baseF.Ed['HubRad']
        Twr2ShftZ   = self.baseF.Ed['Twr2Shft']

        # TODO Maybe some of these should be assinged to self? What self? inputs? or main?
        H_bedplate = self.inputs.HhttZ - (self.inputs.L_drive + HubRad)*tools.sind(ShftTilt) # H_bedplate = 4.875 # TODO ADd to self somehow?
        L_bedplate = (H_bedplate - Twr2ShftZ)/tools.tand(ShftTilt) # L_bedplate = 5 m, X-direction not axial

        # https://wisdem.readthedocs.io/en/master/wisdem/drivetrainse/components.html#bedplate
        # Calculate bedplate points (midline)
        # theta = 0: Nose theta = 90: tower - following right-hand rule - rotation around y positive from nose to tower
        
        xc_mid      = L_bedplate*tools.cosd(45) # Ellipse centerline
        xout_mid    = (L_bedplate + self.inputs.Dtt/2)*tools.cosd(45) # Outer curve. Dtt = Tower top diameter 
        xin_mid     = (L_bedplate - self.inputs.Dtt/2)*tools.cosd(45) # Inner curve.
        zc_mid      = H_bedplate*tools.sind(45)
        zout_mid    = (H_bedplate + self.inputs.NoseOD[0]/2)*tools.sind(45)
        zin_mid     = (H_bedplate - self.inputs.NoseOD[0]/2)*tools.sind(45)

        BdpltOD_mid = np.sqrt((xout_mid - xin_mid)**2 + (zout_mid - zin_mid)**2)

        BedpltJoints = {
            'Stator_Attachment':    {'xyz': [-(L_bedplate + self.inputs.Lgsn*tools.cosd(ShftTilt)), 
                                0, 
                                TowerHt + H_bedplate + self.inputs.Lgsn*tools.sind(ShftTilt)]}, 
            'Stator_Mass':           {'xyz': [self.inputs.Tow2GenStatX, 
                                              self.inputs.Tow2GenStatY,  
                                              TowerHt + self.inputs.Tow2GenStatZ], 
                                'Mass': {'JMass': self.inputs.GenStatMass, 
                                        'JMXX': self.inputs.GenStatMXX, 'JMYY': self.inputs.GenStatMYY, 'JMZZ': self.inputs.GenStatMZZ, 
                                        'JMXY': 0, 'JMXZ': 0, 'JMYZ': 0, 'MCGX': 0, 'MCGY': 0, 'MCGZ': 0}},
            'MB1': {'xyz': [-(L_bedplate + (self.inputs.L2n + self.inputs.L12)*tools.cosd(ShftTilt)), 
                            0, 
                            TowerHt + H_bedplate + (self.inputs.L2n + self.inputs.L12)*tools.sind(ShftTilt)]}, 
            'MB2': {'xyz': [-(L_bedplate + self.inputs.L2n*tools.cosd(ShftTilt)), 
                            0, 
                            TowerHt + H_bedplate + self.inputs.L2n*tools.sind(ShftTilt)]}, 
            'Nose_GenSide': {'xyz': [-L_bedplate, 
                            0, 
                            TowerHt + H_bedplate]}, 
            # 'Nose_RotSide': {'xyz': [-(L_bedplate + self.inputs.Lnose*tools.cosd(ShftTilt)),  # NB! MB1 = Nose_RotSide
            #                 0, 
            #                 TowerHt + H_bedplate + self.inputs.Lnose*tools.sind(ShftTilt)]}, 

            'Bedplt_Mid': {'xyz': [-xc_mid, 
                                0, 
                                TowerHt + zc_mid]}, 
                    }

        # Material Bedplate bottom
        self.BPropSetID+=1
        self.BeamProp.append([self.BPropSetID, self.inputs.BdpltE, self.inputs.BdpltG, self.inputs.BdpltRho, self.inputs.Dtt, self.inputs.BdpltT[0]]) #TODO Check indexing!
        BdpltBot_PropSetID = self.BPropSetID

        self.BPropSetID+=1
        self.BeamProp.append([self.BPropSetID, self.inputs.BdpltE, self.inputs.BdpltG, self.inputs.BdpltRho, BdpltOD_mid, (self.inputs.BdpltT[0] + self.inputs.BdpltT[-1])/2]) #TODO Check indexing!
        BdpltMid_PropSetID = self.BPropSetID

        self.BPropSetID+=1
        self.BeamProp.append([self.BPropSetID, self.inputs.BdpltE, self.inputs.BdpltG, self.inputs.BdpltRho, self.inputs.NoseOD[0], self.inputs.BdpltT[-1]]) #TODO Check indexing!
        BdpltTop_PropSetID = self.BPropSetID
        
        self.BPropSetID+=1
        self.BeamProp.append([self.BPropSetID, self.inputs.NoseE, self.inputs.NoseG, self.inputs.NoseRho, self.inputs.NoseOD[0], self.inputs.NoseT[0]]) #TODO Check indexing!
        NoseGenSide_PropSetID = self.BPropSetID

        # self.BPropSetID+=1
        # self.BeamProp.append([self.BPropSetID, self.inputs.NoseE, self.inputs.NoseG, self.inputs.NoseRho, self.inputs.NoseOD[-1], self.inputs.NoseT[-1]]) #TODO Check indexing!
        # NoseRotSide_PropSetID = self.BPropSetID
        NoseRotSide_PropSetID = NoseGenSide_PropSetID
        print("Warning! Assuming same material properties and geometry along drivetrain nose!")

        # Member between tower and bedplate mid
        print("Bedplate bottom to bedplate mid")
        jnt = BedpltJoints['Bedplt_Mid']
        jntCrds = jnt['xyz']
        jntX, jntY, jntZ = jntCrds[0], jntCrds[1], jntCrds[2]

        self.JointID+=1
        self.Joints.append([self.JointID, jntX, jntY, jntZ, 1, 0.0, 0.0, 0.0, 0.0]) 
        
        self.MemberID+=1
        self.Members.append([self.MemberID, self.twrTopJntID, self.JointID, BdpltBot_PropSetID, BdpltMid_PropSetID , 1, -1]) #MType = 1: Beam
        pipe = self.pipeInpByMemID(self.MemberID, TTZ = 0)
        self.drTrMassProps.append([pipe.CMx, pipe.CMy, pipe.CMz, pipe.mass, pipe.JMXX, pipe.JMYY, pipe.JMZZ]) 

        print("Bedplate mid to nose gen side")
        # Member between bedplate mid and nose generator side
        jnt = BedpltJoints['Nose_GenSide']
        jntCrds = jnt['xyz']
        jntX, jntY, jntZ = jntCrds[0], jntCrds[1], jntCrds[2]

        self.JointID+=1
        self.Joints.append([self.JointID, jntX, jntY, jntZ, 1, 0.0, 0.0, 0.0, 0.0]) 
        
        self.MemberID+=1
        self.Members.append([self.MemberID, self.JointID-1, self.JointID, BdpltMid_PropSetID, BdpltTop_PropSetID , 1, -1]) #MType = 1: Beam
        pipe = self.pipeInpByMemID(self.MemberID, TTZ = 0)
        self.drTrMassProps.append([pipe.CMx, pipe.CMy, pipe.CMz, pipe.mass, pipe.JMXX, pipe.JMYY, pipe.JMZZ]) 

        # Member between nose generator side and stator attachment
        print("Bedplate nose gen side to stator attachment")
        jnt = BedpltJoints['Stator_Attachment']
        jntCrds = jnt['xyz']
        jntX, jntY, jntZ = jntCrds[0], jntCrds[1], jntCrds[2]

        self.JointID+=1
        self.Joints.append([self.JointID, jntX, jntY, jntZ, 1, 0.0, 0.0, 0.0, 0.0]) 
        self.StatorAttachJtID = self.JointID #Connection nodes for the bearings
        
        self.MemberID+=1
        self.Members.append([self.MemberID, self.JointID-1, self.JointID, NoseGenSide_PropSetID, NoseGenSide_PropSetID , 1, -1]) #MType = 1: Beam
        pipe = self.pipeInpByMemID(self.MemberID, TTZ = 0)
        self.drTrMassProps.append([pipe.CMx, pipe.CMy, pipe.CMz, pipe.mass, pipe.JMXX, pipe.JMYY, pipe.JMZZ]) 

        # Member between stator attachment to MB2
        print("Bedplate stator attachment and MB2")
        jnt = BedpltJoints['MB2']
        jntCrds = jnt['xyz']
        jntX, jntY, jntZ = jntCrds[0], jntCrds[1], jntCrds[2]

        self.JointID+=1
        self.Joints.append([self.JointID, jntX, jntY, jntZ, 1, 0.0, 0.0, 0.0, 0.0]) 
        self.MB2BedpltJtID = self.JointID #Connection nodes for the bearings
        
        self.MemberID+=1
        self.Members.append([self.MemberID, self.JointID-1, self.JointID, NoseGenSide_PropSetID, NoseGenSide_PropSetID , 1, -1]) #MType = 1: Beam
        pipe = self.pipeInpByMemID(self.MemberID, TTZ = 0)
        self.drTrMassProps.append([pipe.CMx, pipe.CMy, pipe.CMz, pipe.mass, pipe.JMXX, pipe.JMYY, pipe.JMZZ]) 

        # Member between MB2 and MB1
        print("Bedplate MB2 and MB1")
        jnt = BedpltJoints['MB1']
        jntCrds = jnt['xyz']
        jntX, jntY, jntZ = jntCrds[0], jntCrds[1], jntCrds[2]

        self.JointID+=1
        self.Joints.append([self.JointID, jntX, jntY, jntZ, 1, 0.0, 0.0, 0.0, 0.0]) 
        self.MB1BedpltJtID = self.JointID #Connection nodes for the bearings
        
        self.MemberID+=1
        self.Members.append([self.MemberID, self.JointID-1, self.JointID, NoseGenSide_PropSetID, NoseRotSide_PropSetID , 1, -1]) #MType = 1: Beam
        pipe = self.pipeInpByMemID(self.MemberID, TTZ = 0)
        self.drTrMassProps.append([pipe.CMx, pipe.CMy, pipe.CMz, pipe.mass, pipe.JMXX, pipe.JMYY, pipe.JMZZ]) 

        # Add stator mass
        jnt = BedpltJoints['Stator_Mass']
        jntCrds = jnt['xyz']
        mass = jnt['Mass']
        jntX, jntY, jntZ = jntCrds[0], jntCrds[1], jntCrds[2]

        self.JointID+=1
        self.Joints.append([self.JointID, jntX, jntY, jntZ, 1, 0.0, 0.0, 0.0, 0.0]) 
        
        self.MemberID+=1
        self.Members.append([self.MemberID, self.StatorAttachJtID, self.JointID, self.RPropSetID, self.RPropSetID , 3, -1]) #MType = 3: Rigid link

        self.ConcentratedMasses.append([self.JointID, mass['JMass'], mass['JMXX'], mass['JMYY'], mass['JMZZ'], 
                                                        mass['JMXY'], mass['JMXZ'], mass['JMYZ'], 
                                                        mass['MCGX'], mass['MCGY'], mass['MCGZ']])
        self.drTrMassProps.append([jntX, jntY, jntZ, mass['JMass'], mass['JMXX'], mass['JMYY'], mass['JMZZ']]) 

        # # TODO: Legge til underveis isteden?    
        # Sum up bedplate mass        
        # bdpltLength = BedpltLtot
        # bdpltXArea  = (np.pi/4)*(self.inputs.BedpltOD**2-(self.inputs.BedpltOD-2*self.inputs.BedpltT)**2)
        # bdpltMass = self.inputs.BedpltRho*bdpltLength*bdpltXArea
        # # Uniform mass, assume similar beamprops along the whole shaft
        # bdpltCMx = Gen_x-(Gen_x-MB1_x)/2
        # bdpltCMy = Gen_y-(Gen_y-MB1_y)/2
        # bdpltCMz = Gen_z-(Gen_z-MB1_z)/2
        
        # R_i = (self.inputs.BedpltOD-2*self.inputs.BedpltT)/2
        # R_o = self.inputs.BedpltOD/2
        
        # #Inertia about bedplate COG in global coordinate system
        # bdpltIxx = 1/2*bdpltMass*(R_i**2+R_o**2)
        # bdpltIyy = 1/12*bdpltMass*(3*(R_i**2+R_o**2) + bdpltLength**2)  
        # bdpltIzz = 1/12*bdpltMass*(3*(R_i**2+R_o**2) + bdpltLength**2)  
        
        # self.drTrMassProps.append([bdpltCMx, bdpltCMy, bdpltCMz, bdpltMass, bdpltIxx, bdpltIyy, bdpltIzz]) 

    

    def buildShaft(self):
        """
        Build shaft in SubDyn
        """

        ShftTilt = self.baseF.Ed['ShftTilt']
        Tow2MB1X = self.inputs.Tow2MB1X
        Tow2MB2X = self.inputs.Tow2MB2X
        Tow2ShftZ = self.baseF.Ed['Twr2Shft']
        TowerHeight = self.baseF.Ed['TowerHt']
        Tow2GBSuppX = self.inputs.Tow2GBSuppX
        ShftProps = self.inputs.ShftProps

        Tow2ShftUpstrX  = Tow2MB1X-ShftProps['Shaft tip to MB1']['L']*tools.cosd(ShftTilt)
        Tow2ShftDwnstrX = Tow2MB2X+ShftProps['MB2 to shaft end']['L']*tools.cosd(ShftTilt)

        ShftJoints = {
            'ShftStrt': {'xyz': [Tow2ShftUpstrX, 0, 
                            np.abs(Tow2ShftUpstrX*tools.tand(ShftTilt))+Tow2ShftZ+TowerHeight]},
            'MB1':      {'xyz': [Tow2MB1X, 0, 
                            np.abs(Tow2MB1X*tools.tand(ShftTilt))+Tow2ShftZ+TowerHeight]},
            'MB2':      {'xyz': [Tow2MB2X, 0, 
                            np.abs(Tow2MB2X*tools.tand(ShftTilt))+Tow2ShftZ+TowerHeight]},
            'ShftEnd':  {'xyz': [Tow2ShftDwnstrX, 0, 
                            np.abs(Tow2ShftDwnstrX*tools.sind(ShftTilt))+Tow2ShftZ+TowerHeight]},
            'GB':       {'xyz': [Tow2GBSuppX, 0, 
                            np.abs(Tow2GBSuppX*tools.tand(ShftTilt))+Tow2ShftZ+TowerHeight]}}

        for jntName, val in ShftJoints.items(): 
            jntCrds = val['xyz']
            jntX, jntY, jntZ = jntCrds[0], jntCrds[1], jntCrds[2]
            
            self.JointID += 1
            self.Joints.append([self.JointID, jntX, jntY, jntZ, 1, 0.0, 0.0, 0.0, 0.0]) 
            
            if jntName == 'ShftStrt': #Upwind end of shaft
                ShftTipCoord = [jntX, jntY, jntZ]
                self.ShftTipCoord = ShftTipCoord
                if self.interfaceLocation == 'shaft':
                    self.InterfaceJoints.append([self.JointID, 1, 1, 1, 1, 1, 1])  #All fixities must be set to 1 in the current version of SubDyn
            elif jntName == 'GB': #Rigid link between shaft end and gearbox support
                self.GBShftJtID = self.JointID #Connection nodes for the bearings
                  # No unijoint or pin joints, rigid link between end of flexible shaft and GB
                self.MemberID+=1
                self.Members.append([self.MemberID, self.JointID-1, self.JointID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 3: Rigid link
            # Add beam segments
            else:
                if jntName == 'MB1':
                    MatProps = ShftProps['Shaft tip to MB1']
                    self.BPropSetID+=1
                    self.BeamProp.append([self.BPropSetID, MatProps['E'], MatProps['G'], MatProps['rho'], MatProps['D_o'], MatProps['t']])
                    self.MemberID+=1
                    self.Members.append([self.MemberID, self.JointID-1, self.JointID, self.BPropSetID, self.BPropSetID, 1, -1]) #MType = 1: Beam
                    self.MB1ShftJtID = self.JointID #Connection nodes for the bearings
                    self.MB1ShftMemID = self.MemberID #Member connecting main bearing location along shaft and shaft tip
                elif jntName == 'MB2':
                    MatProps = ShftProps['MB1 to MB2']
                    self.BPropSetID+=1
                    self.BeamProp.append([self.BPropSetID, MatProps['E'], MatProps['G'], MatProps['rho'], MatProps['D_o'], MatProps['t']])
                    self.MemberID+=1
                    self.Members.append([self.MemberID, self.JointID-1, self.JointID, self.BPropSetID, self.BPropSetID, 1, -1]) #MType = 1: Beam
                    self.MB2ShftJtID = self.JointID #Connection nodes for the bearings
                    self.MB2ShftMemID = self.MemberID #Member connecting main bearing location along shaft and shaft tip
                elif jntName == 'ShftEnd':
                    MatProps = ShftProps['MB2 to shaft end']
                    self.BPropSetID+=1
                    self.BeamProp.append([self.BPropSetID, MatProps['E'], MatProps['G'], MatProps['rho'], MatProps['D_o'], MatProps['t']])
                    self.MemberID+=1
                    self.Members.append([self.MemberID, self.JointID-1, self.JointID, self.BPropSetID, self.BPropSetID, 1, -1]) #MType = 1: Beam
                    ShftEndCoord = [jntX, jntY, jntZ]
                    self.ShftEndJtID = self.JointID # End of flexible shaft
                    self.ShftEndMemID = self.MemberID # End of flexible shaft
        
        #--------------Sum up shaft mass 
        shftMass = 0
        for shftSegment, MatProps in ShftProps.items():
            shftLength = MatProps['L']
            shftXArea  = (np.pi/4)*(MatProps['D_o']**2-(MatProps['D_o']-2*MatProps['t'])**2)
            #Assuming uniform non-tapered cross-section
            shftMass += MatProps['rho']*shftLength*shftXArea

        #TODO: Update CM for varying shaft
        shftCMx = ShftEndCoord[0]-(ShftEndCoord[0]-ShftTipCoord[0])/2
        shftCMy = ShftEndCoord[1]-(ShftEndCoord[1]-ShftTipCoord[1])/2
        shftCMz = ShftEndCoord[2]-(ShftEndCoord[2]-ShftTipCoord[2])/2

        R_i = (ShftProps['MB1 to MB2']['D_o']-2*ShftProps['MB1 to MB2']['t'])/2
        R_o = ShftProps['MB1 to MB2']['D_o']/2
        shftIxxLoc = 1/2*shftMass*(R_i**2+R_o**2) #Local coordinate system
        shftIyyLoc = 1/12*shftMass*(3*(R_i**2+R_o**2) + shftLength**2)  #Local coordinate system
        shftIzzLoc = 1/12*shftMass*(3*(R_i**2+R_o**2) + shftLength**2)  #Local coordinate system
        
        #--------------Transforming shaft mass moment of inertia to global coordinate system. This has minimal effect.         
        T = np.array([[     cosd(self.ed['ShftTilt']),      0,          cosd(90+self.ed['ShftTilt'])],
                      [     0,                              1,          0                           ], 
                      [     cosd(90-self.ed['ShftTilt']),   0,          cosd(self.ed['ShftTilt'])   ]])
        
        Iloc = np.array([[  shftIxxLoc,                     0,          0                           ],
                      [     0,                              shftIyyLoc, 0                           ],
                      [     0,                              0,          shftIzzLoc                  ]])
        
        TIloc = np.matmul(T, Iloc)
        TIlocT = np.matmul(TIloc, np.transpose(T))     
        Iglob  = TIlocT

        #Assuming no off-diagonal terms
        shftIxxGlob = Iglob[0][0]
        shftIyyGlob = Iglob[1][1]
        shftIzzGlob = Iglob[2][2]
        self.drTrMassProps.append([shftCMx, shftCMy, shftCMz, shftMass, shftIxxGlob, shftIyyGlob, shftIzzGlob]) 
        
        return
    
    def buildMainBearing(self, MB1or2 = 'MB1'):
        if MB1or2 == 'MB1':
            MBShftJoint = self.jointByID(self.MB1ShftJtID)
            MBShftJtID = self.MB1ShftJtID
            MBBdpltJtID = self.MB1BedpltJtID
            MBSpring = self.inputs.MB1Spring
        elif MB1or2 == 'MB2':
            MBShftJoint = self.jointByID(self.MB2ShftJtID)
            MBShftJtID = self.MB2ShftJtID
            MBBdpltJtID = self.MB2BedpltJtID
            MBSpring = self.inputs.MB2Spring
        else: 
            print('Error: No familiar main bearing names!')
        
        MBShftX = MBShftJoint[1]
        MBShftZ = MBShftJoint[3]

        # Spring
        self.SPropSetID += 1
        self.SpringProp.append([self.SPropSetID, 
                                MBSpring['k11'], 0, 0, 0, 0, 0, #k12, k13, k14, k15, k16
                                MBSpring['k22'], 0, 0, 0, 0, #k23, k24, k25, k26
                                MBSpring['k33'], 0, 0, 0, #k34, k35, k36
                                MBSpring['k44'], 0, 0, #k45, k46
                                MBSpring['k55'], 0, #k56
                                MBSpring['k66']])

        if self.MBCosmID != False: # Already exists
            pass 
        else:
            self.CosmID += 1
            cosms_temp = [self.CosmID] + self.inputs.MB_cosm
            self.Cosms.append(cosms_temp)
            self.MBCosmID = self.CosmID

        self.JointID+=1 # Same position, new joint -> spring of zero length
        self.Joints.append([self.JointID, 
                            MBShftX, 0, MBShftZ,
                            1, 0.0, 0.0, 0.0, 0.0])
        
        self.MemberID+=1
        self.Members.append([self.MemberID, MBShftJtID, self.JointID, self.SPropSetID, self.SPropSetID, 5, self.MBCosmID]) #MType = 5: Spring w/o length
        if MB1or2 == 'MB1':
            self.MB1MemID = self.MemberID
        elif MB1or2 == 'MB2':
            self.MB2MemID = self.MemberID

        #Rigid link to bedplate
        self.MemberID+=1
        self.Members.append([self.MemberID, self.JointID, MBBdpltJtID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 3: Rigid link

    def buildGBox(self):
        """Build gearbox supports and gearbox concentrated mass"""
        GBShftJoint = self.jointByID(self.GBShftJtID)
        
        GBShftX = GBShftJoint[1]
        GBShftZ = GBShftJoint[3]
        
        #-------Cosine matrix-----------------------------#
        if any(lst[1:] == self.inputs.GB_cosm for lst in self.Cosms): # Gearbox cosm = MB cosm
            self.GBCosmID = self.MBCosmID
        else:
            self.CosmID += 1
            cosms_temp = [self.CosmID] + self.inputs.GB_cosm
            self.Cosms.append(cosms_temp)

        #-------Beam properties---------------------------#
        # Stiff beam in y-direction
        L_beam_y = self.inputs.Tow2GBSuppY

        # Spring with zero length
        self.SPropSetID+=1
        self.SpringProp.append([self.SPropSetID, 
                                self.inputs.GBSpring['k11'], 0, 0, 0, 0, 0, #k12, k13, k14, k15, k16
                                self.inputs.GBSpring['k22'], 0, 0, 0, 0, #k23, k24, k25, k26
                                self.inputs.GBSpring['k33'], 0, 0, 0, #k34, k35, k36
                                self.inputs.GBSpring['k44'], 0, 0, #k45, k46
                                self.inputs.GBSpring['k55'], 0, #k56
                                self.inputs.GBSpring['k66']])

        sPropSetID = self.SPropSetID
        sCosmID = self.CosmID
        
        #-------Beam elements---------------------------#

        for y in [L_beam_y, -L_beam_y]: # No pin joints
            #Stiff beam in y-direction
            self.JointID+=1
            self.Joints.append([self.JointID, GBShftX, y, GBShftZ, 1, 0.0, 0.0, 0.0, 0.0])
            self.MemberID+=1
            self.Members.append([self.MemberID, self.GBShftJtID, self.JointID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 3: Rigid link

            #Spring with no length or mass
            self.JointID+=1
            self.Joints.append([self.JointID, GBShftX, y, GBShftZ, 1, 0.0, 0.0, 0.0, 0.0])
            self.MemberID+=1
            self.Members.append([self.MemberID, self.JointID-1, self.JointID, sPropSetID, sPropSetID, 5, sCosmID]) #MType = 5: Spring with diagonal terms
            self.GBSuppMemID.append(self.MemberID)
            
            #Rigid link to bedplate
            self.MemberID+=1
            self.Members.append([self.MemberID, self.JointID, self.GBBdpltJtID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 3: Rigid link

        #Gearbox mass at gearbox center of gravity. Linked to shaft joint between shaft and torque arms

        cgX = self.inputs.Tow2GBMassX
        cgY = self.inputs.Tow2GBMassY
        cgZ = self.inputs.Tow2GBMassZ + self.baseF.Ed['TowerHt']
        self.JointID+=1
        self.Joints.append([self.JointID, cgX, cgY, cgZ, 1, 0.0, 0.0, 0.0, 0.0])
        
        self.MemberID+=1
        self.Members.append([self.MemberID, self.JointID, self.GBShftJtID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 3: Rigid link
        
        self.ConcentratedMasses.append([self.JointID, self.inputs.GBMass, 
                                        self.inputs.GBMXX, self.inputs.GBMYY, self.inputs.GBMZZ,
                                        self.inputs.GBMXY, self.inputs.GBMXZ, self.inputs.GBMYZ, 
                                        0, 0, 0]) #'MCGX', 'MCGY', 'MCGZ'
        
        self.drTrMassProps.append([cgX, cgY, cgZ, 
                                   self.inputs.GBMass, 
                                   self.inputs.GBMXX, self.inputs.GBMYY, self.inputs.GBMZZ]) 

    def nacMassIner(self, add_nacelle_yaw_inertia = False, add_yaw_br_mass = False):
        """Add point mass reflecting additional nacelle mass not accounted for in the drivetrain model (relative to NacMass and NacYiner in original ElastoDyn-files)
            If add_nacelle_yaw_inertia == True: 
                Calculate remaining yaw inertia based on original ElastoDyn-file and added drivetrain inertia.
        """
        
        #---Calculate remaining tower top mass and inertia-----
        #This mass and inertia is not included in subDyn drivetrain
        #Will be added to tower top at correct CM
        print(self.drTrMassProps)
        drTrMassProps = np.array(self.drTrMassProps)
        
        if len(drTrMassProps) == 0:
            print('No drivetrain components with mass added to SubDyn. All of nacelle mass will be contained in a point mass in SubDyn.')
            drTrMass = 0
        else: 
            drTrMass = np.sum(drTrMassProps[:,3])
        nacPtMassM = self.baseF.Ed['NacMass']-drTrMass #remaining mass that needs to be allocated
        #Calculate new CM
        CM_orig = [self.baseF.Ed['NacCMxn'], self.baseF.Ed['NacCMyn'], 
                    self.baseF.Ed['NacCMzn']+self.baseF.Ed['TowerHt']] #NacCMzn is measured from tower top
        NacMass_orig = self.baseF.Ed['NacMass'] #Original ElastoDyn NacMass
        
        NacPtMassCM = []
        CM_tests = []
        for i, CMi in enumerate(CM_orig):  #Loop through CMx, CMy, CMz
            CM_Mtot = CMi*NacMass_orig #Original ED mass times original CM
            CM_test = 0
            for ptMass in drTrMassProps: #Loop through each drivetrain component having a mass
                CM = ptMass[i]
                mass = ptMass[3]
                
                CM_Mtot += -CM*mass
                CM_test += CM*mass
        
            NacPtMassCM.append(CM_Mtot/nacPtMassM)
            CM_tests.append(CM_test)

        assert CM_tests[0]+NacPtMassCM[0]*nacPtMassM == pytest.approx(NacMass_orig*CM_orig[0])
        assert CM_tests[1]+NacPtMassCM[1]*nacPtMassM == pytest.approx(NacMass_orig*CM_orig[1])
        assert CM_tests[2]+NacPtMassCM[2]*nacPtMassM == pytest.approx(NacMass_orig*CM_orig[2])
    
        CMx = NacPtMassCM[0]
        CMy = NacPtMassCM[1]
        CMz = NacPtMassCM[2]
        
        #Zero local inertia
        Jxx = 0
        Jyy = 0
        Jzz = 0
        mass = nacPtMassM
            
        #---Rigid link to point mass-------
    
        self.JointID+=1
        self.Joints.append([self.JointID, CMx, CMy, CMz, 1, 0.0, 0.0, 0.0, 0.0])
        
        self.MemberID+=1
        if self.interfaceLocation == 'tower': 
            self.Members.append([self.MemberID, self.JointID, self.twrTopJntID, self.StiffBeamPropID, self.StiffBeamPropID, 1, -1]) #Cannot have rigid link connected to tower interface joint. Use stiff beam instead. 
        else: 
            self.Members.append([self.MemberID, self.JointID, self.twrTopJntID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 3: Rigid link
        
        self.ConcentratedMasses.append([self.JointID, mass, Jxx, Jyy, Jzz, 0, 0, 0, 0, 0, 0]) 

        # Add nacelle yaw inertia (about tower vertical axis/yaw axis) #TODO CHeck if this is correct!

        
        # Not for DTU 10 MW drivetrain
        # For IEA 15 MW, it should be calculated, based on remaining inertia after drivetrain is modeled
        if add_nacelle_yaw_inertia:
            JzzSum = 0
            for prop in drTrMassProps:
                cgx = prop[0]
                cgy = prop[1]
                cgz = prop[2]
                mass = prop[3]
                Jzz = prop[6] # Relative to cgx, cgy, cgz
                
                # Steiner's theorem to find global mass moments of inertia at X = 0, Y = 0
                center_of_mass = np.array([cgx, cgy, cgz])
                d_squared = np.dot(center_of_mass, center_of_mass)
                
                I_steiner = mass * d_squared * np.eye(3) - mass * np.outer(center_of_mass, center_of_mass)
                I_global_Z0 = Jzz + np.diag(I_steiner)[2] # mass moment of inertia about vertical axis of a mass located at tower top. 
                JzzSum += I_global_Z0
        else: 
            JzzSum = 0
        if add_yaw_br_mass: 
            yawBrMass = self.baseF.Ed['YawBrMass']
        else: 
            yawBrMass = 0
        self.ConcentratedMasses.append([self.twrTopJntID, yawBrMass, 0, 0, self.baseF.Ed['NacYIner']-JzzSum, 0, 0, 0, 0, 0, 0]) 

    def nacMassInerMultiple(self, CMx1 = 0, CMy1 = 0, CMz1 = 118.41925, mass1 = 206798.6796, # USED FOR DEBUGGING ONLY (a while ago)
                        Jxx1 = 183.6198258, Jyy1 = 1094869.005, Jzz1 = 5075985.327, 
                        CMx2 = 0, CMy2 = 0, CMz2 = 118.41925, mass2 = 206798.6796,
                        Jxx2 = 183.6198258, Jyy2 = 1094869.005, Jzz2 = 5075985.327, 
                        CMx3 = 0, CMy3 = 0, CMz3 = 0, mass3 = 0,
                        Jxx3 = 0, Jyy3 = 0, Jzz3 = 0, 
                        CMx4 = 0, CMy4 = 0, CMz4 = 0, mass4 = 0,
                        Jxx4 = 0, Jyy4 = 0, Jzz4 = 0, 
                        CMx5 = 0, CMy5 = 0, CMz5 = 0, mass5 = 0,
                        Jxx5 = 0, Jyy5 = 0, Jzz5 = 0
                        ):
        """Used for debugging - add several point masses"""
        #---Set nacelle mass and inertia to approximately zero in ElastoDyn
        self.ed['NacMass'] = 0 #TODO: Seems like it is okay setting these to zero
        
        self.ed['NacYIner'] = 0 #TODO: Seems like it is okay setting these to zero
        # self.ed['NacYIner'] = self.ed['NacMass']*(self.origEd['NacCMxn']**2+self.origEd['NacCMyn']**2)+1
        self.ed.write(self.edPath)
        
        #TODO: Values are calculated in spreadsheet now because some tweeking was needed to get positive inertias of the point mass. 
        
        #---Rigid link to point mass-------
        
        self.JointID+=1
        self.Joints.append([self.JointID, CMx1, CMy1, CMz1, 1, 0.0, 0.0, 0.0, 0.0])
        
        self.MemberID+=1
        self.Members.append([self.MemberID, self.JointID, self.twrTopJntID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 3: Rigid link
        
        self.ConcentratedMasses.append([self.JointID, mass1, Jxx1, Jyy1, Jzz1, 0, 0, 0, 0, 0, 0])
            
        #2
        self.JointID+=1
        self.Joints.append([self.JointID, CMx2, CMy2, CMz2, 1, 0.0, 0.0, 0.0, 0.0])
        
        self.MemberID+=1
        self.Members.append([self.MemberID, self.JointID, self.twrTopJntID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 3: Rigid link
        
        self.ConcentratedMasses.append([self.JointID, mass2, Jxx2, Jyy2, Jzz2, 0, 0, 0, 0, 0, 0])

        # #3
        self.JointID+=1
        self.Joints.append([self.JointID, CMx3, CMy3, CMz3, 1, 0.0, 0.0, 0.0, 0.0])
        
        self.MemberID+=1
        self.Members.append([self.MemberID, self.JointID, self.twrTopJntID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 3: Rigid link
        
        self.ConcentratedMasses.append([self.JointID, mass3, Jxx3, Jyy3, Jzz3, 0, 0, 0, 0, 0, 0])

        #4
        self.JointID+=1
        self.Joints.append([self.JointID, CMx4, CMy4, CMz4, 1, 0.0, 0.0, 0.0, 0.0])
        
        self.MemberID+=1
        self.Members.append([self.MemberID, self.JointID, self.twrTopJntID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 3: Rigid link
        
        self.ConcentratedMasses.append([self.JointID, mass4, Jxx4, Jyy4, Jzz4, 0, 0, 0, 0, 0, 0])

        #5
        self.JointID+=1
        self.Joints.append([self.JointID, CMx5, CMy5, CMz5, 1, 0.0, 0.0, 0.0, 0.0])
        
        self.MemberID+=1
        self.Members.append([self.MemberID, self.JointID, self.twrTopJntID, self.RPropSetIDMassless, self.RPropSetIDMassless, 3, -1]) #MType = 3: Rigid link
        
        self.ConcentratedMasses.append([self.JointID, mass5, Jxx5, Jyy5, Jzz5, 0, 0, 0, 0, 0, 0])

    def shiftTurbineX(self, x, fstTmplFile, fstTmplPath):
        """Function to shift the whole turbine an offset in x-direction. Not complete, last attempt of analysis failed"""
        #Update file paths in .fst
        self.fst['MooringFile'] = '"../Mooring/DTU_10MW_OO_GoM_MoorDyn_shiftX.dat"'
        self.fst.write(self.fstPath)
        
        #Template files
        templFst                = FASTInputFile(os.path.join(fstTmplPath, fstTmplFile))
        edTmplFile = templFst['EDFile'][1:-1]
        edTmplPath = os.path.join(fstTmplPath, edTmplFile)
        edTmpl  = FASTInputFile(edTmplPath)
        
        sdTmplFile = templFst['SubFile'][1:-1]
        sdTmplPath = os.path.join(fstTmplPath, sdTmplFile)
        sdTmpl  = FASTInputFile(sdTmplPath)

        hdTmplFile = templFst['HydroFile'][1:-1]
        hdTmplPath = os.path.join(fstTmplPath, hdTmplFile)
        hdTmpl  = FASTInputFile(hdTmplPath)

        iwTmplFile = templFst['InflowFile'][1:-1]
        iwTmplPath = os.path.join(fstTmplPath, iwTmplFile)
        iwTmpl  = FASTInputFile(iwTmplPath)

        #Shift x-coordinates in SubDyn
        self.sd['Joints'] = np.array(self.sd['Joints'])
        self.sd['Joints'][:,1] = np.array(sdTmpl['Joints'])[:, 1] + x
        self.sd.write(self.sdPath)
        #Written to SubDyn inpute file in "write"-function 
        
        #Shift x-coordinates in ElastoDyn 
        self.ed['TowerHt'] = np.round(edTmpl['TowerHt'] - x*tand(edTmpl['ShftTilt']),2)
        self.ed['TowerBsHt'] = np.round(edTmpl['TowerBsHt'] - x*tand(edTmpl['ShftTilt']),2)
        self.ed['PtfmRefzt'] = np.round(edTmpl['PtfmRefzt'] - x*tand(edTmpl['ShftTilt']),2)
        self.ed['OverHang'] = np.round(edTmpl['OverHang'] + x, 2)
        self.ed['ShftGagL'] = np.round(edTmpl['ShftGagL'] + x, 2)
        self.ed['NacCMxn'] = np.round(edTmpl['NacCMxn'] + x, 2)
        self.ed['NcIMUxn'] = np.round(edTmpl['NcIMUxn'] + x, 2)
        self.ed['PtfmCMxt'] = np.round(edTmpl['PtfmCMxt'] + x, 2)
        self.ed.write(self.edPath)
        
        #Shift x-coordinates in HydroDYn
        self.hd['Joints'][:,1]= hdTmpl['Joints'][:,1] + x
        self.hd['PtfmRefxt'] = hdTmpl['PtfmRefxt'] + x
        self.hd['WaveHs'] = 0.0
        self.hd['WaveMod'] = 0
        
        #Shift wind box in 
        self.iw['XOffset'] = iwTmpl['XOffset']+x
        self.iw['WindVxiList'] = iwTmpl['WindVxiList']+x
        self.iw.write(self.iwPath)
    
    def outputs(self):
        """HydroDyn and SubDyn outputs"""
        #HydroDyn outputs - body motions and wave elevation
        if self.platformType == 'floating': 
            self.hd['---------------------- OUTPUT'] = ['', '"Wave1Elev"               - Wave elevation at the platform reference point (0,  0)', 
                                                        '"B1Surge, B1Sway, B1Heave, B1Roll, B1Pitch, B1Yaw" - Platform motion']
            
    
        #TODO: Rather just output all member nodes!
        #Add tower top and tower base to member outputs
        self.memOutDict = {} # keep track of member outputs for post-processing -> written to JSON-file
        
        self.sd['MemberOuts']   = []
        memOut_count = 0
        #--------Tower Base-----------------#
        if self.platformType == 'monopile':
            #--------Monopile Base Member--------#
            self.sd['MemberOuts'].append(np.array([self.pileBsMemID, 2, 1, 2])) #Pile Base
            memOut_count += 1
            self.memOutDict['PileBas'] = {}
            self.memOutDict['PileBas']['memCount'] = [memOut_count]
            self.memOutDict['PileBas']['ID'] = self.pileBsMemID
        else:
            #--------Tower Base Member--------#
            self.sd['MemberOuts'].append(np.array([self.twrBasMemID, 2, 1, 2])) #Tower base member with two nodes
            memOut_count = 1
            self.memOutDict['TwrBas'] = {}
            self.memOutDict['TwrBas']['memCount'] = [memOut_count]
            self.memOutDict['TwrBas']['ID'] = self.twrBasMemID

        #--------Tower Top------------------#
        self.sd['MemberOuts'].append(np.array([self.twrTopMemID, 2, 1, 2])) #Tower top member with two nodes
        memOut_count += 1
        self.memOutDict['TwrTop'] = {}
        self.memOutDict['TwrTop']['memCount'] = [memOut_count]
        self.memOutDict['TwrTop']['ID'] = self.twrTopMemID

        try: 
            #--------Shaft Tip------------------#
            self.sd['MemberOuts'].append(np.array([self.MB1ShftMemID, 2, 1, 2])) #Shaft tip to main bearing connection along shaft
            memOut_count += 1
            self.memOutDict['ShftTip'] = {}
            self.memOutDict['ShftTip']['memCount'] = [memOut_count]
            self.memOutDict['ShftTip']['ID'] = self.MB1ShftMemID
            
            self.memOutDict['Shaft'] = {}
            self.memOutDict['Shaft']['memCount'] = [memOut_count] #Add shaft MB1 and MB2 later on
            self.memOutDict['Shaft']['ID'] = []   #We only care about the MN - outputs, not MJ, for this particular part
        except:
            print("WARNING when printing member outputs: 'SubDyn' object has no attribute 'MB1ShftMemID'")

        try:
            #--------MB1------------------#
            self.sd['MemberOuts'].append(np.array([self.MB1MemID, 2, 1, 2])) 
            memOut_count += 1
            self.memOutDict['MB1'] = {}
            self.memOutDict['MB1']['memCount'] = [memOut_count]
            self.memOutDict['MB1']['ID'] = self.MB1MemID
        except:
            print("WARNING when printing member outputs: 'SubDyn' object has no attribute 'MB1MemID'")
        try:
            #--------MB2------------------#
            self.sd['MemberOuts'].append(np.array([self.MB2MemID, 2, 1, 2])) 
            memOut_count += 1
            self.memOutDict['MB2'] = {}
            self.memOutDict['MB2']['memCount'] = [memOut_count]
            self.memOutDict['MB2']['ID'] = self.MB2MemID
        except:
            print("WARNING when printing member outputs: 'SubDyn' object has no attribute 'MB2MemID'")

            #--------GB support------------------# #TODO: Add a try here??
        for i, ID in enumerate(self.GBSuppMemID):
            self.sd['MemberOuts'].append(np.array([ID, 2, 1, 2])) #Tower top member with two nodes
            memOut_count += 1
            self.memOutDict['GBSupp' + str(i+1)] = {}
            self.memOutDict['GBSupp' + str(i+1)]['memCount'] = [memOut_count]
            self.memOutDict['GBSupp' + str(i+1)]['ID'] = ID
        
        #--------Shaft mid (MB1 to MB2) member------------------#
        try: 
            self.sd['MemberOuts'].append(np.array([self.MB2ShftMemID, 2, 1, 2])) #Shaft tip to main bearing connection along shaft
            memOut_count += 1
            try: 
                self.memOutDict['Shaft']['memCount'].append(memOut_count) 
            except:
                self.memOutDict['Shaft']['memCount'] = [memOut_count] #If MB1 not added
                print("Warning: Seems like shaft tip to MB1 is not added to 'shaft'-part in JSON-file")
        except:
            print("WARNING when printing member outputs: 'SubDyn' object has no attribute 'MB2ShftMemID'")
        
        #--------Shaft end (MB2 to GBsupp) member------------------#
        try: 
            self.sd['MemberOuts'].append(np.array([self.ShftEndMemID, 2, 1, 2])) #Shaft tip to main bearing connection along shaft
            memOut_count += 1
            try: 
                self.memOutDict['Shaft']['memCount'].append(memOut_count) 
            except:
                self.memOutDict['Shaft']['memCount'] = [memOut_count] #If MB1 not added
                print("Warning: Seems like shaft tip to MB1 is not added to 'shaft'-part in JSON-file")
        except:
            print("WARNING when printing member outputs: 'SubDyn' object has no attribute 'ShftEndMemID'")

        self.sd['NMOutputs'] = len(self.sd['MemberOuts']) 
        #TODO: I think this is updated
        if self.sd['NMOutputs']>9:
            print('WARNING: Only 9 lines are allowd in the SDOutlist')
        
        #Update SDOutlist
        for i, dictionary in enumerate(self.sd.data):
            if type(dictionary['value']) == str and 'SDOutList' in dictionary['value']:
                sdOutList_istart = i+1
            if type(dictionary['value']) == str and 'END' in dictionary['value']:
                sdOutList_iend = i
        
        del self.sd.data[sdOutList_istart:sdOutList_iend] 
        # TODO Check that output is not hard-coded
        sdoutlist = [{'value': '"M1N1TDxss, M1N1TDyss, M1N1TDzss, M2N2TDxss, M2N2TDyss, M2N2TDzss" \t\t - Tower base and tower top displacements', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M1N1RDxe, M1N1RDye, M1N1RDze, M2N2RDxe, M2N2RDye, M2N2RDze" \t\t - Tower base and tower top rotations (local coords)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M2N2TAxe, M2N2TAye, M2N2TAze, M2N2RAxe, M2N2RAye, M2N2RAze" \t\t - Tower base and tower top accelerations (local coords)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M1N1FKxe, M1N1FKye, M1N1FKze, M2N2FKxe, M2N2FKye, M2N2FKze" \t\t - Static (elastic) forces (tower base and tower top)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M1N1MKxe, M1N1MKye, M1N1MKze, M2N2MKxe, M2N2MKye, M2N2MKze" \t\t - Static (elastic) moments (tower base and tower top)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M1N1FMxe, M1N1FMye, M1N1FMze, M2N2FMxe, M2N2FMye, M2N2FMze" \t\t - Dynamic (inertia) forces (tower base and tower top)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M1N1MMxe, M1N1MMye, M1N1MMze, M2N2MMxe, M2N2MMye, M2N2MMze" \t\t - Dynamic (inertia) moments (tower base and tower top)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M3N1TDxss, M3N1TDyss, M3N1TDzss, M3N1RDxe, M3N1RDye, M3N1RDze" \t\t - Shaft tip motion', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M3N2TDxss, M3N2TDyss, M3N2TDzss, M3N2RDxe, M3N2RDye, M3N2RDze" \t\t - Shaft tip motion', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M3N1TAxe, M3N1TAye, M3N1TAze, M3N1RAxe, M3N1RAye, M3N1RAze" \t\t - Shaft accelerations (local coords)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M3N2TAxe, M3N2TAye, M3N2TAze, M3N2RAxe, M3N2RAye, M3N2RAze" \t\t - Shaft accelerations (local coords)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M3N1FMxe, M3N1FMye, M3N1FMze, M3N2FMxe, M3N2FMye, M3N2FMze" \t\t - Dynamic (inertia) forces (shaft to MB1)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M3N1MMxe, M3N1MMye, M3N1MMze, M3N2MMxe, M3N2MMye, M3N2MMze" \t\t - Dynamic (inertia) moments (shaft to MB1)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M3N1FKxe, M3N1FKye, M3N1FKze, M3N2FKxe, M3N2FKye, M3N2FKze" \t\t - Static (elastic) forces (shaft to MB1)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M3N1MKxe, M3N1MKye, M3N1MKze, M3N2MKxe, M3N2MKye, M3N2MKze" \t\t - Static (elastic) moments (shaft to MB1)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M4N1TDxss, M4N1TDyss, M4N1TDzss, M4N1RDxe, M4N1RDye, M4N1RDze" \t\t - MB1 flex beam motion', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M4N2TDxss, M4N2TDyss, M4N2TDzss, M4N2RDxe, M4N2RDye, M4N2RDze" \t\t - MB1 flex beam motion', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M4N1TAxe, M4N1TAye, M4N1TAze, M4N1RAxe, M4N1RAye, M4N1RAze" \t\t - MB1 accelerations (local coords)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M4N2TAxe, M4N2TAye, M4N2TAze, M4N2RAxe, M4N2RAye, M4N2RAze" \t\t - MB1 accelerations (local coords)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M4N1FMxe, M4N1FMye, M4N1FMze, M4N2FMxe, M4N2FMye, M4N2FMze" \t\t - Dynamic (inertia) forces (MB1 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M4N1MMxe, M4N1MMye, M4N1MMze, M4N2MMxe, M4N2MMye, M4N2MMze" \t\t - Dynamic (inertia) moments (MB1 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M4N1FKxe, M4N1FKye, M4N1FKze, M4N2FKxe, M4N2FKye, M4N2FKze" \t\t - Static (elastic) forces (MB1 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M4N1MKxe, M4N1MKye, M4N1MKze, M4N2MKxe, M4N2MKye, M4N2MKze" \t\t - Static (elastic) moments (MB1 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M5N1TDxss, M5N1TDyss, M5N1TDzss, M5N1RDxe, M5N1RDye, M5N1RDze" \t\t - MB2 flex beam motion', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M5N2TDxss, M5N2TDyss, M5N2TDzss, M5N2RDxe, M5N2RDye, M5N2RDze" \t\t - MB2 flex beam motion', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M5N1FMxe, M5N1FMye, M5N1FMze, M5N2FMxe, M5N2FMye, M5N2FMze" \t\t - Dynamic (inertia) forces (MB2 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M5N1MMxe, M5N1MMye, M5N1MMze, M5N2MMxe, M5N2MMye, M5N2MMze" \t\t - Dynamic (inertia) moments (MB2 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M5N1FKxe, M5N1FKye, M5N1FKze, M5N2FKxe, M5N2FKye, M5N2FKze" \t\t - Static (elastic) forces (MB2 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M5N1MKxe, M5N1MKye, M5N1MKze, M5N2MKxe, M5N2MKye, M5N2MKze" \t\t - Static (elastic) moments (MB2 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M6N1TDxss, M6N1TDyss, M6N1TDzss, M6N1RDxe, M6N1RDye, M6N1RDze" \t\t - GBSupp1 flex beam motion', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M6N2TDxss, M6N2TDyss, M6N2TDzss, M6N2RDxe, M6N2RDye, M6N2RDze" \t\t - GBSupp1 flex beam motion', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M6N1FMxe, M6N1FMye, M6N1FMze, M6N2FMxe, M6N2FMye, M6N2FMze" \t\t - Dynamic (inertia) forces (GBSupp1 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M6N1MMxe, M6N1MMye, M6N1MMze, M6N2MMxe, M6N2MMye, M6N2MMze" \t\t - Dynamic (inertia) moments (GBSupp1 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M6N1FKxe, M6N1FKye, M6N1FKze, M6N2FKxe, M6N2FKye, M6N2FKze" \t\t - Static (elastic) forces (GBSupp1 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M6N1MKxe, M6N1MKye, M6N1MKze, M6N2MKxe, M6N2MKye, M6N2MKze" \t\t - Static (elastic) moments (GBSupp1 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M7N1TDxss, M7N1TDyss, M7N1TDzss, M7N1RDxe, M7N1RDye, M7N1RDze" \t\t - GBSupp2 flex beam motion', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M7N2TDxss, M7N2TDyss, M7N2TDzss, M7N2RDxe, M7N2RDye, M7N2RDze" \t\t - GBSupp2 flex beam motion', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M7N1FMxe, M7N1FMye, M7N1FMze, M7N2FMxe, M7N2FMye, M7N2FMze" \t\t - Dynamic (inertia) forces (GBSupp2 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0}, 
                     {'value': '"M7N1MMxe, M7N1MMye, M7N1MMze, M7N2MMxe, M7N2MMye, M7N2MMze" \t\t - Dynamic (inertia) moments (GBSupp2 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M7N1FKxe, M7N1FKye, M7N1FKze, M7N2FKxe, M7N2FKye, M7N2FKze" \t\t - Static (elastic) forces (GBSupp2 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M7N1MKxe, M7N1MKye, M7N1MKze, M7N2MKxe, M7N2MKye, M7N2MKze" \t\t - Static (elastic) moments (GBSupp2 flex beam)', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M8N1TDxss, M8N1TDyss, M8N1TDzss, M8N1RDxe, M8N1RDye, M8N1RDze" \t\t - Shaft pt2 flex beam motion', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M8N2TDxss, M8N2TDyss, M8N2TDzss, M8N2RDxe, M8N2RDye, M8N2RDze" \t\t - Shaft pt2 flex beam motion', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M9N1TDxss, M9N1TDyss, M9N1TDzss, M9N1RDxe, M9N1RDye, M9N1RDze" \t\t - Shaft pt3 flex beam motion', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"M9N2TDxss, M9N2TDyss, M9N2TDzss, M9N2RDxe, M9N2RDye, M9N2RDze" \t\t - Shaft pt3 flex beam motion', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"ReactFXss, ReactFYss, ReactFZss, ReactMXss, ReactMYss, ReactMZss" \t\t - - Base moments and forces for bottom-fixed/landbased.', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"IntfFXss, IntfFYss, IntfFZss, IntfMXss, IntfMYss, IntfMZss" \t\t - - Interface joint moments and forces.', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"IntfTDXss, IntfTDYss, IntfTDZss, IntfRDXss, IntfRDYss , IntfRDZss" \t\t - - Displacements and rotations of the TP reference point in global coordinate sys.', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     {'value': '"IntfTAXss, IntfTAYss, IntfTAZss, IntfRAXss, IntfRAYss, IntfRAZss" \t\t - - Translational and rotational accelerations of the TP reference point (platform reference point) location in SS coordinate system', 'label': '', 'isComment': True, 'descr': '', 'tabType': 0},
                     ]
        
        self.sd.data[sdOutList_istart:sdOutList_istart] = sdoutlist

    def outputsTestCase(self, NSegments = 8):

        self.sd['MemberOuts']   = []
        i = 1
        for seg in range(1, NSegments+1):
            self.sd['MemberOuts'].append(np.array([seg, 2, 1, 2])) 
        
        self.sd['NMOutputs'] = len(self.sd['MemberOuts'])
        if self.sd['NMOutputs']>9:
            print('WARNING: Only 9 lines are allowd in the SDOutlist')
        
        #Update SDOutlist
        for i, dictionary in enumerate(self.sd.data):
            if type(dictionary['value']) == str and 'SDOutList' in dictionary['value']:
                sdOutList_istart = i+1
            if type(dictionary['value']) == str and 'END' in dictionary['value']:
                sdOutList_iend = i
        
        del self.sd.data[sdOutList_istart:sdOutList_iend]
        sdoutlist = []
        segPars = ['TDxss', 'FKxe', 'MKxe']
        for seg in range(1, NSegments+1):
            for par in segPars:
                for jt in range(1, 3):
                    text = '"'      
                    text += 'M' + str(seg) + 'N' + str(jt) + par + ', '
                    par = par.replace('x', 'y')
                    text += 'M' + str(seg) + 'N' + str(jt) + par + ', '
                    par = par.replace('y', 'z')
                    text += 'M' + str(seg) + 'N' + str(jt) + par + ', '
                    par = par.replace('z', 'x')
                    text = text.strip(', ')
                    text += '"'
                    dict = {'value': text,
                            'label': '',
                            'isComment': True, 
                            'descr': '', 
                            'tabType': 0}
                    sdoutlist.append(dict)

        reactPars = ['ReactFXss', 'IntfFXss', 'IntfTDXss']
        for par in reactPars: 
            text = '"'      
            text += par + ', '
            par = par.replace('X', 'Y')
            text += par + ', '
            par = par.replace('Y', 'Z')
            text += par + ', '
            if 'IntfT' in par:
                par = par.replace('Z', 'X').replace('TD', 'RD')
            else:
                par = par.replace('Z', 'X').replace('F', 'M')
            text += par + ', '
            par = par.replace('X', 'Y')
            text += par + ', '
            par = par.replace('Y', 'Z')
            text += par + ', '
            text = text.strip(', ')
            text += '"'
            dict = {'value': text,
                    'label': '',
                    'isComment': True, 
                    'descr': '', 
                    'tabType': 0}
            sdoutlist.append(dict)

        self.sd.data[sdOutList_istart:sdOutList_istart] = sdoutlist

    def writeJSON(self):
        "Write JSON-file labelling important subdyn-members. Useful for post-processing"
        memberLabels = {}
        for part, content in self.memOutDict.items():
            memberLabels[part] = {}

            counts = self.memOutDict[part]['memCount']
            if part == 'TwrBas':
                memberLabels[part]['MemOut']      = ['M' + str(counts[0]) + 'N1']
            elif part == 'PileBas':
                memberLabels[part]['MemOut']      = ['M' + str(counts[0]) + 'N1']
            elif part == 'TwrTop':
                memberLabels[part]['MemOut']      = ['M' + str(counts[0]) + 'N2']
            else:
                memberLabels[part]['MemOut']      = list(np.array([['M' + str(count) + 'N1', 'M' + str(count) + 'N2'] for count in counts]).flatten())

            ID = self.memOutDict[part]['ID']
            if ID != []:
                # if part in ['TwrBas', 'ShftTip']: # Keep node J1 only
                #     memberLabels[part]['JointOut']    = ['M' + str(ID) + 'J1']
                # elif part in ['TwrTop', 'MB1', 'MB2', 'GBSupp1', 'GBSupp2']: # Keep node J2 only
                #     memberLabels[part]['JointOut']    = ['M' + str(ID) + 'J2']
                # else:
                memberLabels[part]['JointOut']    = ['M' + str(ID) + 'J1', 'M' + str(ID) + 'J2']
            else: 
                memberLabels[part]['JointOut']    = ID #TODO: Shaft - we don't want the jointouts here for now - works better with plot script (but maybe later)
        
        # Serializing json
        json_object = json.dumps(memberLabels, indent=4)
        
        # Writing to sample.json
        with open(os.path.join(self.mainPath, 'memberLabels.json'), "w") as outfile:
            outfile.write(json_object)  

    def writeHydroDyn(self):
        self.hd.write(self.hdPath)

    def writeFst(self):
        """Update .fst-file"""
        #Turn on SubDyn calculation in FAST-file
        self.fst['CompSub'] = 1
        self.fst['NumCrctn'] = 1 
        self.fst['DT_Out'] = 'default' # TODO Change!
            
        self.fst.write(self.mainF.FstPath) 


    def writeElastoDyn(self):
        """Various changes in ElastoDyn-file"""
        smallValue = 0.01
        Tow2ShftZ  = self.baseF.Ed['Twr2Shft']  #This is higher than the Simpack-model. Otherwise use input Tow2ShftZ

        #----- DOFs-------#
        self.ed['TeetDOF'] = False
        self.ed['DrTrDOF'] = True
        self.ed['GenDOF'] = True
        self.ed['YawDOF'] = False
        
        self.ed['TwFADOF1'] = False
        self.ed['TwFADOF2'] = False
        self.ed['TwSSDOF1'] = False
        self.ed['TwSSDOF2'] = False

        self.ed['PtfmSgDOF'] = True
        self.ed['PtfmSwDOF'] = True
        self.ed['PtfmHvDOF'] = True
        self.ed['PtfmRDOF'] = True
        self.ed['PtfmPDOF'] = True
        self.ed['PtfmYDOF'] = True

        #----- Initial conditions-------#
        self.ed['NacYaw']   = 0
        self.ed['TTDspFA']  = 0
        self.ed['TTDspSS']  = 0
        # Ptfm initial displacements are referring to the tower top for bottom-fixed turbines and to platform displacements, with reference to the tower top, for floating turbines. 
        self.ed['PtfmSurge']    = 0
        self.ed['PtfmSway']     = 0
        self.ed['PtfmHeave']    = 0
        self.ed['PtfmRoll']     = 0
        self.ed['PtfmPitch']    = 0
        self.ed['PtfmYaw']      = 0

        #----- Turbine Configuration-------#
        self.ed['Twr2Shft'] = smallValue
        self.ed['TowerHt'] = np.round(self.baseF.Ed['TowerHt']+Tow2ShftZ-smallValue, 2)
        self.ed['TowerBsHt'] = np.round(self.baseF.Ed['TowerHt']+Tow2ShftZ-smallValue*2,2)
        self.ed['PtfmRefzt'] = np.round(self.baseF.Ed['TowerHt']+Tow2ShftZ-smallValue*2,2)
        self.ed['PtfmCMzt'] = self.ed['PtfmRefzt']
        
        # #TODO: Fix
        # if not ShftTipCoordX:
        #     #Calculates the position of the shaft strain gauge in ElastoDyn so that it coincides with the shaft tip in SubDyn
        #     self.ed['ShftGagL'] = np.round(-(self.ed['OverHang']-self.ShftTipCoord[0]/cosd(-self.ed['ShftTilt'])),3)
        # else: 
        #     self.ed['ShftGagL'] = np.round(-(self.ed['OverHang']-ShftTipCoordX/cosd(-self.ed['ShftTilt'])),3)

        #----- Mass and inertia -----#
        self.ed['NacMass'] = 0 
        self.ed['NacYIner'] = 0 
        self.ed['YawBrMass'] = 0 
        self.ed['PtfmRIner'] = self.inputs.PtfmRIner
        self.ed['PtfmMass'] = 0.0
        self.ed['PtfmPIner'] = 0.0
        self.ed['PtfmYIner'] = 0.0

        #----- Tower ------#        
        self.ed['TwrNodes'] = 1

        #----- Blades -----#

        #----- Drivetrain -----#
        if self.inputs.DTTorDmp:
            self.ed['DTTorDmp'] = self.inputs.DTTorDmp
        else: 
            self.ed['DTTorDmp'] = self.baseF.Ed['DTTorDmp']

        #----- Output ------#
        self.ed['NTwGages'] = 0
        self.ed['TwrGagNd'] = 0

        self.ed["OutList"] = [out_param for out_param in self.ed["OutList"]
                      if out_param.split('-')[0][1:4] not in ['TwH', 'Yaw', 'Twr', 'Ptf']]
        
        #----- Write ElastoDyn-file ------#
        self.ed.write(self.edPath)

    def writeElastoDynTestCase(self, NacMass=200000, NacCMxn = 3, NacCMzn = 0, HubMass=10, Tow2ShftZ = 0, TowHt = 100):
        """Various changes in ElastoDyn-file"""
        smallValue = 0.1

        #----- DOFs-------#
        self.ed['FlapDOF1'] = False
        self.ed['FlapDOF2'] = False
        self.ed['EdgeDOF'] = False
        self.ed['TeetDOF'] = False
        self.ed['DrTrDOF'] = False
        self.ed['GenDOF'] = False
        self.ed['YawDOF'] = False

        self.ed['TwFADOF1'] = False
        self.ed['TwFADOF2'] = False
        self.ed['TwSSDOF1'] = False
        self.ed['TwSSDOF2'] = False

        self.ed['PtfmSgDOF'] = True
        self.ed['PtfmSwDOF'] = False
        self.ed['PtfmHvDOF'] = True
        self.ed['PtfmRDOF'] = False
        self.ed['PtfmPDOF'] = True
        self.ed['PtfmYDOF'] = False
        
        #----- Turbine Configuration-------#
        self.ed['TipRad'] = 0.1
        self.ed['HubRad'] = 0
        self.ed['PreCone(1)'] = 0
        self.ed['PreCone(2)'] = 0
        self.ed['PreCone(3)'] = 0
        self.ed['OverHang'] = 0
        self.ed['ShftGagL'] = 0
        self.ed['ShftTilt'] = 0

        self.ed['NacCMxn'] = NacCMxn
        self.ed['NacCMzn'] = NacCMzn
        self.ed['NcIMUxn'] = 0
        self.ed['NcIMUzn'] = 0
    
        self.ed['Twr2Shft'] = Tow2ShftZ
        self.ed['TowerHt'] = np.round(TowHt+smallValue,4)
        self.ed['TowerBsHt'] = np.round(TowHt,4)
        self.ed['PtfmRefzt'] = np.round(TowHt,4)
        self.ed['PtfmCMzt'] = 0.0

        #----- Initial conditions -----#
        self.ed['PtfmSurge'] = 0.0
        self.ed['PtfmSway'] = 0.0
        self.ed['PtfmHeave'] = 0.0
        self.ed['PtfmRoll'] = 0.0
        self.ed['PtfmPitch'] = 0.0
        self.ed['PtfmYaw'] = 0.0

        #----- Mass and inertia -----#
        self.ed['HubMass'] = 0
        self.ed['HubIner'] = 0
        self.ed['GenIner'] = 0
        self.ed['NacMass'] = NacMass
        self.ed['NacYIner'] = 1800000

        self.ed['PtfmMass'] = 0.0
        self.ed['PtfmRIner'] = 0.0
        self.ed['PtfmPIner'] = 1000
        self.ed['PtfmYIner'] = 0.0
        
        #----- Blade -----#
        self.ed['BldNodes'] = 3
        self.ed['BldFile(1)'] = '"ED_Blade_Light.dat"'
        self.ed['BldFile(2)'] = '"ED_Blade_Light.dat"'
        self.ed['BldFile(3)'] = '"ED_Blade_Light.dat"'

        #----- Drivetrain -----#
        self.ed['GBRatio'] = 1
        self.ed['DTTorSpr'] = 0
        self.ed['DTTorDmp'] = 0

        #----- Tower ------#
        self.ed['TwrNodes'] = 3
        self.ed['TwrFile'] = '"ED_Tower_Light.dat"'

        #----- Output -----#
        self.ed['NTwGages'] = 1
        self.ed['TwrGagNd'] = 3

        if not any('"TwHt1FLxt"' in mystring for mystring in self.ed['OutList']):
            self.ed['OutList'].append('"TwHt1FLxt" 						-')
        if not any('"TwHt1FLyt"' in mystring for mystring in self.ed['OutList']):
            self.ed['OutList'].append('"TwHt1FLyt" 						-')
        if not any('"TwHt1FLzt"' in mystring for mystring in self.ed['OutList']):
            self.ed['OutList'].append('"TwHt1FLzt" 						-')    
        if not any('"TwHt1MLxt"' in mystring for mystring in self.ed['OutList']):
            self.ed['OutList'].append('"TwHt1MLxt" 						-')    
        if not any('"TwHt1MLyt"' in mystring for mystring in self.ed['OutList']):
            self.ed['OutList'].append('"TwHt1MLyt" 						-')    
        if not any('"TwHt1MLzt"' in mystring for mystring in self.ed['OutList']):
            self.ed['OutList'].append('"TwHt1MLzt" 						-')    

        #Write ElastoDyn-file
        self.ed.write(self.edPath)

    def writeSubDyn(self):
        # Test before write
        print('Writing subdyn...............')
        self.assertions

        self.sd['FEMMod']   = 1 #Only Euler-Bernoulli beams supported so far
        self.sd['NDiv']     = 1 
        
        #Check that IDs are unique
        self.sd['Members']      = self.Members
        self.sd['BeamProp']     = np.array(self.BeamProp)
        self.sd['Joints']       = self.Joints
        self.sd['RigidProp']    = self.RigidProp
        self.sd['SpringProp']   = self.SpringProp
        self.sd['BaseJoints']   = self.BaseReactionJoints #No reaction joints for floating substructure
        self.sd['ConcentratedMasses']   = self.ConcentratedMasses
        self.sd['InterfaceJoints']      = self.InterfaceJoints
        # self.sd['MemberOuts']   = [] #No outputs atm
        self.sd['MemberCosineMatrix'] = self.Cosms

        self.sd['OutCBModes']   =  1
        self.sd['OutAll']       = True #All nodal forces and moments are output
        self.sd['CBMod']        = True #Perform C-B-reduction?
        self.sd['SttcSolve']    = True 
        self.sd['GuyanDampMod'] =  1 #Use Rayleigh damping
        self.sd['RayleighDamp'] = self.inputs.RayleighCoffs
        
        if self.platformType == 'floating':
            self.sd['GuyanLoadCorrection']   =  True
        else: # Only verified without GLC for monopile and landbased
            self.sd['GuyanLoadCorrection']  =  False

        # User defined inputs
        self.sd['Nmodes']       = self.inputs.NModes #Number of CB-modes
        self.sd['JDampings']    = self.inputs.CBDamp
        
        self.sd.write(self.sdPath)

    def writeModel(self):
        """Wrapper for updating fast input files"""
        self.writeFst()
        self.writeElastoDyn()
        self.writeSubDyn()

        if hasattr(self, 'hd'):
            self.writeHydroDyn()

def rod_prop(s, Di, ti, rho):
    L = s.max() - s.min()

    def equal_pts(xi):
        if len(xi) < len(s) and len(xi) == 2:
            x = np.interp((s - s.min()) / L, [0, 1], xi)
        elif len(xi) == len(s):
            x = xi
        else:
            raise ValueError("Unknown grid of input", str(xi))
        return x

    D = equal_pts(Di)
    print("D", D)
    t = equal_pts(ti)
    print("t", t)
    y = 0.25 * rho * np.pi * (D**2 - (D - 2 * t) ** 2)
    m = np.trapz(y, s)
    cm = np.trapz(y * s, s) / m
    Dm = D.mean()
    tm = t.mean()
    I = np.array(
        [
            0.5 * 0.25 * (Dm**2 + (Dm - 2 * tm) ** 2),
            (1.0 / 12.0) * (3 * 0.25 * (Dm**2 + (Dm - 2 * tm) ** 2) + L**2),
            (1.0 / 12.0) * (3 * 0.25 * (Dm**2 + (Dm - 2 * tm) ** 2) + L**2),
        ]
    )
    return m, cm, m * I

# --- MAIN SCRIPT STARTS BELOW: ----------------------------------------------#
if __name__ == '__main__':

    # --- INPUT VARIABLES AND CONSTANTS --------------------------------------#

    # --- SCRIPT CONTENT -----------------------------------------------------#
    # initiate model
    # # IEA 15 MW floating - # TODO: put in separate script
    # mymodel = Model(platform_type='floating', interface_location='tower')
    # mymodel.timeStep()
    # mymodel.floatingPlatform()
    # mymodel.buildTower(tower_type = 'tower_from_wisdem')
    # mymodel.nacMassIner(add_nacelle_yaw_inertia = True) #TODO:rewise input
    # mymodel.outputs()
    # mymodel.writeModel()
    # mymodel.writeJSON()

    # # DTU 10 MW landbased
    # mymodel = Model(platform_type='landbased', interface_location='shaft')
    # mymodel.timeStep()
    # mymodel.onshoreFixedBase()
    # mymodel.buildTower(tower_type = 'tower_from_elastodyn')
    # mymodel.buildBedplate(bedplate_type = 'straight_beam')
    # mymodel.buildShaft()
    # mymodel.buildMainBearing(MB1or2 = 'MB1')
    # mymodel.buildMainBearing(MB1or2 = 'MB2')
    # mymodel.buildGBox()
    # mymodel.nacMassIner(add_nacelle_yaw_inertia = False) 
    # mymodel.outputs()
    # mymodel.writeModel()
    # mymodel.writeJSON()
    
    # DTU 10 MW monopile
    mymodel = Model(platform_type='monopile', interface_location='shaft')
    mymodel.timeStep()
    mymodel.monoPile()
    mymodel.buildTower(tower_type = 'tower_from_elastodyn')
    mymodel.buildBedplate(bedplate_type = 'straight_beam')
    mymodel.buildShaft()
    mymodel.buildMainBearing(MB1or2 = 'MB1')
    mymodel.buildMainBearing(MB1or2 = 'MB2')
    mymodel.buildGBox()
    mymodel.nacMassIner(add_nacelle_yaw_inertia = False) 
    mymodel.outputs()
    mymodel.writeModel()
    mymodel.writeJSON()