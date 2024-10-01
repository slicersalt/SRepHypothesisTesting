import csv
import json
import logging
import numpy as np
import os
import shutil
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *

from cpns.principal_nested_spheres import PNS

try:
    from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
except ModuleNotFoundError:
    slicer.util.pip_install('astropy')
    from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

try:
  from tsquared import HotellingT2
except ModuleNotFoundError:
  slicer.util.pip_install('tsquared')
  from tsquared import HotellingT2


#
# SRepHypothesisTesting
#

class SRepHypothesisTesting(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "SRep Hypothesis Testing"
    self.parent.categories = ["Skeleton, topology"]
    self.parent.dependencies = []
    self.parent.contributors = ["David Allemang (Kitware)"]
    self.parent.helpText = (
      "Perform hypothesis testing on an s-rep population"
    )
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = (
      "This file was originally developed by Jean-Christophe Fillion-Robin, "
      "Kitware Inc., Andras Lasso, PerkLab, and Steve Pieper, Isomics, Inc. "
      "and was partially funded by NIH grant 3P41RR013218-12S1."
    )


#
# SRepHypothesisTestingWidget
#

class SRepHypothesisTestingWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.logic = None

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer)
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/SRepHypothesisTesting.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    self.ui.inputCSV.nameFilters += ("*.csv",)

    self.logic = SRepHypothesisTestingLogic()
    self.logic.templatePath = self.resourcePath('reordered_srep.vtk')

    self.ui.ApplyButton.connect('clicked(bool)', self.onApplyButton)

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    try:
      self.logic.run(
        self.ui.inputCSV.currentPath
      )
    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: {}".format(e))
      import traceback
      traceback.print_exc()


# Helper classes/functions
class Spoke(object):
    def __init__(self, radius=None, direction=None,
                 base_pt=None, absolute_uv=None, bdry_pt=None):
        self.r = radius
        self.U = np.array(direction, dtype=np.float64)
        self.p = np.array(base_pt, dtype=np.float64)
        self.coords = absolute_uv # coords in (u, v) coords system, (offset_r, offset_c, r0, c0)
        self.ext = np.inf       # extension (i.e., boundary to the medial axis of between-object)
        self.link_dir = None

        ## This is predefined threshold. Only links that are smaller than this threshold are considered
        self.delta_min = 0.5 ## distance from z to z_prime, where z is the end of the extension of this spoke
        self.ext_pt = None
        
        if bdry_pt is not None:
            ## compute r, U from base_pt and bdry_pt
            assert base_pt is not None, "Need both the skeletal point and bdry point"
            s = np.array(bdry_pt) - base_pt
            self.r = np.linalg.norm(s)
            self.U = s / self.r

    def scale(self, scale=1):
        # if np.isinf(self.ext):
        #     return None
        # if scale < 1: scale += 1
        return Spoke(scale * self.r, self.U, self.p)

    def getB(self):
        return self.p + self.r * self.U

class Frame:
    def __init__(self, x, v1, v2, v3):
        # construct the frame here
        self.x = x
        self.v1 = v1 / np.linalg.norm(v1)
        self.v2 = v2 / np.linalg.norm(v2)
        self.v3 = v3 / np.linalg.norm(v3)
    
    def as_matrix(self):
        return np.column_stack((self.v1, self.v2, self.v3))
    
    def in_basis(self,new_basis):
        # Change this frame to new_basis and return
        W = new_basis.as_matrix()
        return Frame(self.x,
                     np.linalg.solve(W,self.v1),
                     np.linalg.solve(W,self.v2),
                     np.linalg.solve(W,self.v3))

def theta_tau_to_linear(theta, tau1, tau2 = 0):
    if tau2 > 0:
        tau2 = np.ceil(tau2)
    if theta < 0: # Wrap around
        theta = theta + 24
    if theta > 23: # Wrap around
        theta = theta % 24
    return int(3*theta + 2*tau1 + 72*tau2)

def crest_to_linear(theta):
    if theta > 23:
        theta = theta % 24
    return 144 + theta

def is_spine_extension(linear):
    return(linear in [0,1,2,36,37,38,72,73,74,108,109,110])
    
def is_crest(linear):
    return (linear > 143)

def is_repeated(linear):
    return (linear in [39,42,45,48,51,54,57,60,63,66,69,111,114,117,120,123,126,129,132,135,138,141,144])

def get_skins(top_spokes, bot_spokes, crs_spokes,levels = [0.0,0.25,0.5,0.75,1.0]):
    skins = []
    for level in levels:
        skin = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        cells = vtk.vtkCellArray()

        # Build quads
        # Top
        for i in range(len(top_spokes)):
            top_pt = top_spokes[i].scale(level).getB()
            pts.InsertNextPoint(top_pt)
        for tau2 in [0]:
            for theta in range(24):
                for tau1 in [0,0.5]:
                    linear = theta_tau_to_linear(theta,tau1,tau2)
                    linear_theta_p = theta_tau_to_linear(theta+1,tau1,tau2)
                    linear_tau1_p = theta_tau_to_linear(theta,tau1+0.5,tau2)
                    linear_both_p = theta_tau_to_linear(theta+1,tau1+0.5,tau2)

                    area = vtk.vtkTriangle().TriangleArea(pts.GetPoint(linear),pts.GetPoint(linear_theta_p),pts.GetPoint(linear_tau1_p))
                    if area > 0.00001:
                        cell = vtk.vtkTriangle()
                        cell.GetPointIds().SetId(0,linear)
                        cell.GetPointIds().SetId(1,linear_theta_p)
                        cell.GetPointIds().SetId(2,linear_tau1_p)
                        # cell.GetPointIds().SetId(3,linear_tau1_p)
                        cellid = cells.InsertNextCell(cell)
                        # print(f"{theta},{tau1}: {linear} {linear_theta_p} {linear_tau1_p}")

                    area = vtk.vtkTriangle().TriangleArea(pts.GetPoint(linear_theta_p),pts.GetPoint(linear_tau1_p),pts.GetPoint(linear_both_p))
                    if area > 0.00001:                        
                        cell2 = vtk.vtkTriangle()
                        cell2.GetPointIds().SetId(0,linear_theta_p)
                        cell2.GetPointIds().SetId(1,linear_tau1_p)
                        cell2.GetPointIds().SetId(2,linear_both_p)
                        cellid = cells.InsertNextCell(cell2)
                    # print(f"{theta},{tau1}: {linear_theta_p} {linear_tau1_p} {linear_both_p}")

        # Bot
        for i in range(len(bot_spokes)):
            bot_pt = bot_spokes[i].scale(level).getB()
            pts.InsertNextPoint(bot_pt)
        for tau2 in [1]:
            for theta in range(24):
                for tau1 in [0,0.5]:
                    linear = theta_tau_to_linear(theta,tau1,tau2)
                    linear_theta_p = theta_tau_to_linear(theta+1,tau1,tau2)
                    linear_tau1_p = theta_tau_to_linear(theta,tau1+0.5,tau2)
                    linear_both_p = theta_tau_to_linear(theta+1,tau1+0.5,tau2)

                    area = vtk.vtkTriangle().TriangleArea(pts.GetPoint(linear),pts.GetPoint(linear_theta_p),pts.GetPoint(linear_tau1_p))
                    if area > 0.001:
                        cell = vtk.vtkTriangle()
                        cell.GetPointIds().SetId(0,linear)
                        cell.GetPointIds().SetId(1,linear_theta_p)
                        cell.GetPointIds().SetId(2,linear_tau1_p)
                        # cell.GetPointIds().SetId(3,linear_tau1_p)
                        cellid = cells.InsertNextCell(cell)
                        # print(f"{theta},{tau1}: {linear} {linear_theta_p} {linear_tau1_p}")

                    area = vtk.vtkTriangle().TriangleArea(pts.GetPoint(linear_theta_p),pts.GetPoint(linear_tau1_p),pts.GetPoint(linear_both_p))
                    if area > 0.001:
                        cell2 = vtk.vtkTriangle()
                        cell2.GetPointIds().SetId(0,linear_theta_p)
                        cell2.GetPointIds().SetId(1,linear_tau1_p)
                        cell2.GetPointIds().SetId(2,linear_both_p)
                        cellid = cells.InsertNextCell(cell2)
                        # print(f"{theta},{tau1}: {linear_theta_p} {linear_tau1_p} {linear_both_p}")

        # Crest
        for i in range(len(crs_spokes)):
            crs_pt = crs_spokes[i].scale(level).getB()
            pts.InsertNextPoint(crs_pt)
        for theta in range(24):
            linear = crest_to_linear(theta)
            next_linear = crest_to_linear(theta+1)

            # Connect to top
            top_linear = theta_tau_to_linear(theta,1,0)
            top_linear_next = theta_tau_to_linear(theta+1,1,0)

            area = vtk.vtkTriangle().TriangleArea(pts.GetPoint(linear),pts.GetPoint(top_linear),pts.GetPoint(top_linear_next))
            if area > 0.001:
                cell = vtk.vtkTriangle()
                cell.GetPointIds().SetId(0,linear)
                cell.GetPointIds().SetId(1,top_linear)
                cell.GetPointIds().SetId(2,top_linear_next)
                cellid = cells.InsertNextCell(cell)

                
            area = vtk.vtkTriangle().TriangleArea(pts.GetPoint(linear),pts.GetPoint(top_linear_next),pts.GetPoint(next_linear))
            if area > 0.001:
                cell.GetPointIds().SetId(0,linear)
                cell.GetPointIds().SetId(1,top_linear_next)
                cell.GetPointIds().SetId(2,next_linear)
                cellid = cells.InsertNextCell(cell)

            # Connect to bot
            bot_linear = theta_tau_to_linear(theta,1,1)
            bot_linear_next = theta_tau_to_linear(theta+1,1,1)
            
            area = vtk.vtkTriangle().TriangleArea(pts.GetPoint(linear),pts.GetPoint(bot_linear),pts.GetPoint(bot_linear_next))
            if area > 0.001:
                cell = vtk.vtkTriangle()
                cell.GetPointIds().SetId(0,linear)
                cell.GetPointIds().SetId(1,bot_linear)
                cell.GetPointIds().SetId(2,bot_linear_next)
                cellid = cells.InsertNextCell(cell)

            area = vtk.vtkTriangle().TriangleArea(pts.GetPoint(linear),pts.GetPoint(bot_linear_next),pts.GetPoint(next_linear))
            if area > 0.001:
                cell.GetPointIds().SetId(0,linear)
                cell.GetPointIds().SetId(1,bot_linear_next)
                cell.GetPointIds().SetId(2,next_linear)
                cellid = cells.InsertNextCell(cell)
        skin.SetPoints(pts)
        skin.SetPolys(cells)

        skins.append(skin)
    return skins

#
# SRepHypothesisTestingLogic
#

class SRepHypothesisTestingLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  
  def json_to_spoke_format(self,data,template):    
    skel = data['EllipticalSRep']['Skeleton']
    srep_pd = vtk.vtkPolyData()
    srep_pd.DeepCopy(template)
    
    num_steps = data['EllipticalSRep']['Steps']

    for i in range(len(skel)):
      s2 = skel[i]
      for j in range(len(s2)):
        spu = np.array(s2[j]['UpSpoke']['SkeletalPoint']['Value'])
        du = np.array(s2[j]['UpSpoke']['Direction']['Value'])

        spd = np.array(s2[j]['DownSpoke']['SkeletalPoint']['Value'])
        dd = np.array(s2[j]['DownSpoke']['Direction']['Value'])


        srep_pd.GetPoints().SetPoint(3*2*i + 2*j,spu)
        srep_pd.GetPoints().SetPoint(3*2*i + 2*j+1,spu+du)

        srep_pd.GetPoints().SetPoint(144 + 3*2*i + 2*j,spd)
        srep_pd.GetPoints().SetPoint(144 + 3*2*i + 2*j+1,spd+dd)

        if j == num_steps:
          spc = np.array(s2[j]['CrestSpoke']['SkeletalPoint']['Value'])
          dc = np.array(s2[j]['CrestSpoke']['Direction']['Value'])
          srep_pd.GetPoints().SetPoint(288 + 2*i,spc)
          srep_pd.GetPoints().SetPoint(288 + 2*i+1,spc+dc)

    return srep_pd

  def build_features(self, srep_poly):
    # setup default features
    num_spokes = srep_poly.GetNumberOfPoints() // 2
    radii = np.zeros((1, num_spokes))
    dirs = np.zeros((3, num_spokes))
    skeletal_pts = np.zeros((3, num_spokes))
    for i in range(num_spokes):
        base_pt_id = i * 2
        skeletal_pt = np.array(srep_poly.GetPoint(base_pt_id))
        skeletal_pts[:, i] = skeletal_pt

        bdry_pt = np.array(srep_poly.GetPoint(base_pt_id + 1))
        radius = np.linalg.norm(bdry_pt - skeletal_pt)
        radii[:, i] = radius
        direction = (bdry_pt - skeletal_pt) / radius
        dirs[:, i] = direction

    # Fix numpy formatting
    skeletal_pts = np.array(skeletal_pts)
    dirs = np.array(dirs)
    radii = radii[0]

    # Build spoke arrays
    top_spokes = []
    bot_spokes = []
    crs_spokes = []

    pd = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    # Top
    for tau2 in [0]:
        for theta in range(24):
            for tau1 in [0,0.5,1]:
                linear = theta_tau_to_linear(theta,tau1,tau2)
                sp = Spoke( radii[linear], dirs[:,linear], skeletal_pts[:,linear] )
                top_spokes.append(sp)

    # Bottom
    for tau2 in [1]:
        for theta in range(24):
            for tau1 in [0,0.5,1]:
                linear = theta_tau_to_linear(theta,tau1,tau2)
                sp = Spoke( radii[linear], dirs[:,linear], skeletal_pts[:,linear] )
                bot_spokes.append(sp)

    # Crest
    for tau2 in [2]:
        for theta in range(24):
            linear = crest_to_linear(theta)
            sp = Spoke( radii[linear], dirs[:,linear], skeletal_pts[:,linear] )
            crs_spokes.append(sp)

    # Build onion skins
    levels = [0.0,0.25,0.5,0.75,1.0]
    skins = get_skins(top_spokes, bot_spokes, crs_spokes, levels)

    skin_normals = []
    for skin in skins:
        norms = vtk.vtkPolyDataNormals()
        norms.SetInputData(skin)
        norms.AutoOrientNormalsOn()
        norms.Update()
        skin_normals.append(norms.GetOutput().GetPointData().GetArray("Normals"))

    pd2 = vtk.vtkPolyData()
    pts2 = vtk.vtkPoints()
    cells2 = vtk.vtkCellArray()

    frames = [[],[],[],[],[]]

    out_pts = vtk.vtkPoints()
    out_pd = vtk.vtkPolyData()

    # Build fitted frames for interior points
    for tau2 in [0,1]:
        for theta in range(24):
            for tau1 in [0,0.5,1]:
                for level in levels:
                    linear = theta_tau_to_linear(theta,tau1,tau2)
                    base_pt = skeletal_pts[:,linear] + level*radii[linear]*dirs[:,linear]
                    if is_spine_extension(linear):
                        # First, compute theta direction
                        if tau1 == 0:
                            theta_plus = theta_tau_to_linear(theta,tau1+0.5,tau2)
                            theta_minus = theta_tau_to_linear(theta,tau1,tau2)
                        elif tau1 == 0.5:
                            theta_plus = theta_tau_to_linear(theta,tau1+0.5,tau2)
                            theta_minus = theta_tau_to_linear(theta,tau1-0.5,tau2)
                        else:
                            theta_plus = theta_tau_to_linear(theta,tau1,tau2)
                            theta_minus = theta_tau_to_linear(theta,tau1-0.5,tau2)
                            
                        plus_pt = skeletal_pts[:,theta_plus] + level*radii[theta_plus]*dirs[:,theta_plus]
                        minus_pt = skeletal_pts[:,theta_minus] + level*radii[theta_minus]*dirs[:,theta_minus]
                        theta_dir = (plus_pt - minus_pt) / 2
                        theta_dir = theta_dir / np.linalg.norm(theta_dir)

                        # Next, compute normal to onion skin
                        if level > 0:
                            normal = dirs[:,linear]
                        else:
                            normals = skin_normals[levels.index(level)]
                            normal = np.array(normals.GetTuple(linear))

                        # Third vector is cross product of first two
                        v = np.cross(theta_dir,normal)

                        frames[levels.index(level)].append(Frame(base_pt,theta_dir,normal,v))
                    else:
                        # First, compute theta direction
                        theta_plus = theta_tau_to_linear(theta+1,tau1,tau2)
                        theta_minus = theta_tau_to_linear(theta-1,tau1,tau2)

                        plus_pt = skeletal_pts[:,theta_plus] + level*radii[theta_plus]*dirs[:,theta_plus]
                        minus_pt = skeletal_pts[:,theta_minus] + level*radii[theta_minus]*dirs[:,theta_minus]
                        theta_dir = (plus_pt - minus_pt) / 2
                        theta_dir = theta_dir / np.linalg.norm(theta_dir)

                        # Next, compute normal to onion skin
                        if level > 0:
                            normal = dirs[:,linear]
                        else:
                            normals = skin_normals[levels.index(level)]
                            normal = np.array(normals.GetTuple(linear))

                        # Third vector is cross product of first two
                        v = np.cross(theta_dir,normal)
                        frames[levels.index(level)].append(Frame(base_pt,theta_dir,normal,v))

    # Build frames for crest
    for theta in range(24):
        for level in levels:
            linear = crest_to_linear(theta)
            base_pt = skeletal_pts[:,linear] + level*radii[linear]*dirs[:,linear]

            theta_plus = crest_to_linear(theta + 1)
            theta_minus = crest_to_linear(theta - 1)
            plus_pt = skeletal_pts[:,theta_plus] + level*radii[theta_plus]*dirs[:,theta_plus]
            minus_pt = skeletal_pts[:,theta_minus] + level*radii[theta_minus]*dirs[:,theta_minus]
            theta_dir = (plus_pt - minus_pt) / 2
            theta_dir = theta_dir / np.linalg.norm(theta_dir)

            # normal direction is just crest spoke direction
            normal = dirs[:,linear]

            v = np.cross(theta_dir,normal)

            frames[levels.index(level)].append(Frame(base_pt,theta_dir,normal,v))

    # Build features from frames
    features = []
    # Start with interior points
    # Features are
    #   Location x change in the theta direction (length and direction)
    #   Location x change in the tau1 direction (length and direction)
    #   Curvatures in the theta direction, i.e., rotations per unit distance (amount and axis)
    #   Curvatures in the tau1 direction, i.e., rotations per unit distance (amount and axis)
    #   Northside spoke (length and direction)
    #   Northside spoke’s frame curvature between skeleton and boundary
    #   Southside spoke (length and direction)						
    #   Southside spoke’s frame curvature between skeleton and boundary
    for theta in range(24):
        for tau1 in [0,0.5,1]:
            curr_linear = theta_tau_to_linear(theta, tau1)
            curr_frame = frames[levels.index(0)][curr_linear]
            if is_spine_extension(curr_linear) or is_repeated(curr_linear):
                continue

            # Location x change in the theta direction (length and direction)
            theta_p1 = theta_tau_to_linear(theta+1,tau1)
            frame_theta_p1 = frames[levels.index(0)][theta_p1]
            theta_m1 = theta_tau_to_linear(theta-1,tau1)
            frame_theta_m1 = frames[levels.index(0)][theta_m1]

            if (is_spine_extension(theta_p1)): # Use backward difference
                v = curr_frame.x - frame_theta_m1.x 
            elif (is_spine_extension(theta_m1)):  # Use forward difference
                v = frame_theta_p1.x - curr_frame.x 
            else:
                v = (frame_theta_p1.x - frame_theta_m1.x) / 2 
            r, lat, lon = cartesian_to_spherical(v[0],v[1],v[2])
            features = features + [lat.value, lon.value, r.value] 

            # Location x change in the tau1 direction (length and direction)
            tau1_p1 = theta_tau_to_linear(theta,tau1+0.5)
            frame_tau1_p1 = frames[levels.index(0)][tau1_p1]

            tau1_m1 = theta_tau_to_linear(theta,tau1-0.5)
            frame_tau1_m1 = frames[levels.index(0)][tau1_m1]

            if tau1 == 0: # Forward difference
                v = frame_tau1_p1.x - curr_frame.x
            elif tau1 == 1: # Backward difference
                v = curr_frame.x - frame_tau1_m1.x
            else: # Central difference
                v = (frame_tau1_p1.x - frame_tau1_m1.x) / 2
            r, lat, lon = cartesian_to_spherical(v[0],v[1],v[2])
            features = features + [lat.value, lon.value, r.value]
            # print(features)

            # Curvatures in the theta direction, i.e., rotations per unit distance
            frame_theta_p1_cob = frame_theta_p1.in_basis(curr_frame)
            frame_theta_m1_cob = frame_theta_m1.in_basis(curr_frame)
            if (is_spine_extension(theta_p1)): # Use backward difference
                rot, rssd = R.align_vectors(curr_frame.as_matrix().T, frame_theta_m1_cob.as_matrix().T)
                d = np.linalg.norm(curr_frame.x - frame_theta_m1_cob.x)
            elif (is_spine_extension(theta_m1)):  # Use forward difference
                rot, rssd = R.align_vectors(frame_theta_p1_cob.as_matrix().T, curr_frame.as_matrix().T)
                d = np.linalg.norm(frame_theta_p1_cob.x - curr_frame.x)
            else:
                rot, rssd = R.align_vectors(frame_theta_p1_cob.as_matrix().T, frame_theta_m1_cob.as_matrix().T)
                d = np.linalg.norm(frame_theta_p1_cob.x - frame_theta_m1_cob.x)
            rotvec = rot.as_rotvec()
            r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
            features = features + [lat.value, lon.value, r.value/d]

            # Curvatures in the tau1 direction, i.e., rotations per unit distance
            frame_tau1_p1_cob = frame_tau1_p1.in_basis(curr_frame)
            frame_tau1_m1_cob = frame_tau1_m1.in_basis(curr_frame)
            if tau1 == 0: # Forward diff
                rot, rssd = R.align_vectors(frame_tau1_p1_cob.as_matrix().T, curr_frame.as_matrix().T)
                d = np.linalg.norm(frame_tau1_p1_cob.x - curr_frame.x)
            elif tau1 == 1: # Backward diff
                rot, rssd = R.align_vectors(curr_frame.as_matrix().T, frame_tau1_m1_cob.as_matrix().T)
                d = np.linalg.norm(curr_frame.x - frame_tau1_m1_cob.x)
            else:
                rot, rssd = R.align_vectors(frame_tau1_p1_cob.as_matrix().T, frame_tau1_m1_cob.as_matrix().T)
                d = np.linalg.norm(frame_tau1_p1_cob.x - frame_tau1_m1_cob.x)
            rotvec = rot.as_rotvec()
            r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
            features = features + [lat.value, lon.value, r.value/d]

            # Northside spoke (length and direction)
            # No need for a new linear index since north is the "default"
            north_spoke = radii[curr_linear]*dirs[:,curr_linear]
            r, lat, lon = cartesian_to_spherical(north_spoke[0],north_spoke[1],north_spoke[2])
            features = features + [lat.value, lon.value, r.value]

            # Northside spoke’s frame curvature between skeleton and boundary
            north_spoke_frame = frames[levels.index(1)][curr_linear]
            north_spoke_frame_cob = north_spoke_frame.in_basis(curr_frame)
            rot, rssd = R.align_vectors(north_spoke_frame_cob.as_matrix().T, curr_frame.as_matrix().T)
            rotvec = rot.as_rotvec()
            r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
            features = features + [lat.value, lon.value, r.value/radii[curr_linear]]

            # Southside spoke (length and direction)	
            # Need a new index for south with tau2 = 1
            south_spoke_linear = theta_tau_to_linear(theta,tau1,1)
            south_spoke = radii[south_spoke_linear]*dirs[:,south_spoke_linear]
            r, lat, lon = cartesian_to_spherical(south_spoke[0],south_spoke[1],south_spoke[2])
            features = features + [lat.value, lon.value, r.value]

            # Southside spoke’s frame curvature between skeleton and boundary
            south_spoke_frame = frames[levels.index(1)][south_spoke_linear]
            south_spoke_frame_cob = south_spoke_frame.in_basis(curr_frame)
            rot, rssd = R.align_vectors(south_spoke_frame_cob.as_matrix().T, curr_frame.as_matrix().T)
            rotvec = rot.as_rotvec()
            r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
            features = features + [lat.value, lon.value, r.value/radii[south_spoke_linear]]

    # (61-6) points * 24 features
    assert len(features) == 55*24

    # Next are fold points
    # Features are
    #   Location x change in the theta direction (length and direction)
    #   Curvatures in the theta direction, i.e., rotations per unit distance
    #   Spoke (length and direction)
    #   Spoke’s frame curvature between skeleton and boundary
    for theta in range(24):
        curr_linear = crest_to_linear(theta)
        curr_frame = frames[0][curr_linear]

        # Location x change in the theta direction (length and direction)
        theta_plus = crest_to_linear(theta + 1)
        frame_plus = frames[0][theta_plus]
        theta_minus = crest_to_linear(theta - 1)
        frame_minus = frames[0][theta_minus]
        v = (frame_plus.x - frame_minus.x) / 2
        r, lat, lon = cartesian_to_spherical(v[0],v[1],v[2])
        features = features + [lat.value, lon.value, r.value] 
        
        # Curvatures in the theta direction, i.e., rotations per unit distance
        frame_plus_cob = frame_plus.in_basis(curr_frame)
        frame_minus_cob = frame_minus.in_basis(curr_frame)
        rot, rssd = R.align_vectors(frame_plus_cob.as_matrix().T, frame_minus_cob.as_matrix().T)
        d = np.linalg.norm(frame_plus_cob.x - frame_minus_cob.x)
        rotvec = rot.as_rotvec()
        r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
        features = features + [lat.value, lon.value, r.value/d]

        # Spoke (length and direction)
        spoke = radii[curr_linear]*dirs[:,curr_linear]
        r, lat, lon = cartesian_to_spherical(spoke[0],spoke[1],spoke[2])
        features = features + [lat.value, lon.value, r.value]

        # Spoke’s frame curvature between skeleton and boundary
        spoke_frame = frames[levels.index(1)][curr_linear]
        spoke_frame_cob = spoke_frame.in_basis(curr_frame)
        rot, rssd = R.align_vectors(spoke_frame_cob.as_matrix().T, curr_frame.as_matrix().T)
        rotvec = rot.as_rotvec()
        r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
        features = features + [lat.value, lon.value, r.value/radii[curr_linear]]

    # Previous + 24 points * 12 features
    assert len(features) == 55*24 + 24*12

    # Next are spine end/extensions
    # Features are
    #   Location x change in the tau1 direction (length and direction)
    #   Curvatures in the tau1 direction, i.e., rotations per unit distance
    #   Northside spoke (length and direction)
    #   Northside spoke’s frame curvature between skeleton and boundary
    #   Southside spoke (length and direction)
    #   Southside spoke’s frame curvature between skeleton and boundary
    for theta in range(24):
        for tau1 in [0,0.5,1]:
            curr_linear = theta_tau_to_linear(theta, tau1)
            curr_frame = frames[levels.index(0)][curr_linear]
            if not is_spine_extension(curr_linear):
                continue

            # Location x change in the tau1 direction (length and direction)
            tau1_p1 = theta_tau_to_linear(theta,tau1+0.5)
            frame_tau1_p1 = frames[levels.index(0)][tau1_p1]

            tau1_m1 = theta_tau_to_linear(theta,tau1-0.5)
            frame_tau1_m1 = frames[levels.index(0)][tau1_m1]

            if tau1 == 0:
                v = frame_tau1_p1.x - curr_frame.x
            elif tau1 == 1:
                v = curr_frame.x - frame_tau1_m1.x
            else:
                v = (frame_tau1_p1.x - frame_tau1_m1.x) / 2
            r, lat, lon = cartesian_to_spherical(v[0],v[1],v[2])
            features = features + [lat.value, lon.value, r.value]

            # Curvatures in the tau1 direction, i.e., rotations per unit distance
            frame_tau1_p1_cob = frame_tau1_p1.in_basis(curr_frame)
            frame_tau1_m1_cob = frame_tau1_m1.in_basis(curr_frame)
            if tau1 == 0: # Forward diff
                rot, rssd = R.align_vectors(frame_tau1_p1_cob.as_matrix().T, curr_frame.as_matrix().T)
                d = np.linalg.norm(frame_tau1_p1_cob.x - curr_frame.x)
            elif tau1 == 1: # Backward diff
                rot, rssd = R.align_vectors(curr_frame.as_matrix().T, frame_tau1_m1_cob.as_matrix().T)
                d = np.linalg.norm(curr_frame.x - frame_tau1_m1_cob.x)
            else:
                rot, rssd = R.align_vectors(frame_tau1_p1_cob.as_matrix().T, frame_tau1_m1_cob.as_matrix().T)
                d = np.linalg.norm(frame_tau1_p1_cob.x - frame_tau1_m1_cob.x)
            rotvec = rot.as_rotvec()
            r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
            features = features + [lat.value, lon.value, r.value/d]

            # Northside spoke (length and direction)
            # No need for a new linear index since north is the "default"
            north_spoke = radii[curr_linear]*dirs[:,curr_linear]
            r, lat, lon = cartesian_to_spherical(north_spoke[0],north_spoke[1],north_spoke[2])
            features = features + [lat.value, lon.value, r.value]

            # # Northside spoke’s frame curvature between skeleton and boundary
            north_spoke_frame = frames[levels.index(1)][curr_linear]
            north_spoke_frame_cob = north_spoke_frame.in_basis(curr_frame)
            rot, rssd = R.align_vectors(north_spoke_frame_cob.as_matrix().T, curr_frame.as_matrix().T)
            rotvec = rot.as_rotvec()
            r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
            features = features + [lat.value, lon.value, r.value/radii[curr_linear]]

            # # Southside spoke (length and direction)	
            # # Need a new index for south with tau2 = 1
            south_spoke_linear = theta_tau_to_linear(theta,tau1,1)
            south_spoke = radii[south_spoke_linear]*dirs[:,south_spoke_linear]
            r, lat, lon = cartesian_to_spherical(south_spoke[0],south_spoke[1],south_spoke[2])
            features = features + [lat.value, lon.value, r.value]

            # # Southside spoke’s frame curvature between skeleton and boundary
            south_spoke_frame = frames[levels.index(1)][south_spoke_linear]
            south_spoke_frame_cob = south_spoke_frame.in_basis(curr_frame)
            rot, rssd = R.align_vectors(south_spoke_frame_cob.as_matrix().T, curr_frame.as_matrix().T)
            rotvec = rot.as_rotvec()
            r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
            features = features + [lat.value, lon.value, r.value/radii[south_spoke_linear]]


    # Previous + 6 points * 18 features
    assert len(features) == 55*24 + 24*12 + 6*18

    # Next are interior onionskin points on spike ends/extensions
    # Features are
    #   Location x change in the tau1 direction (length and direction)
    #   Curvatures in the tau1 direction, i.e., rotations per unit distance
    for theta in range(24):
        for tau1 in [0,0.5,1]:
            for tau2 in [0,1]:
                for level in [0.25,0.5,0.75]:
                    curr_linear = theta_tau_to_linear(theta,tau1,tau2)
                    curr_frame = frames[levels.index(level)][curr_linear]
                    if not is_spine_extension(curr_linear):
                        continue

                    # Location x change in the tau1 direction (length and direction)
                    tau1_p1 = theta_tau_to_linear(theta,tau1+0.5)
                    frame_tau1_p1 = frames[levels.index(level)][tau1_p1]

                    tau1_m1 = theta_tau_to_linear(theta,tau1-0.5)
                    frame_tau1_m1 = frames[levels.index(level)][tau1_m1]

                    if tau1 == 0:
                        v = frame_tau1_p1.x - curr_frame.x
                    elif tau1 == 1:
                        v = curr_frame.x - frame_tau1_m1.x
                    else:
                        v = (frame_tau1_p1.x - frame_tau1_m1.x) / 2
                    r, lat, lon = cartesian_to_spherical(v[0],v[1],v[2])
                    features = features + [lat.value, lon.value, r.value]

                    # Curvatures in the tau1 direction, i.e., rotations per unit distance
                    frame_tau1_p1_cob = frame_tau1_p1.in_basis(curr_frame)
                    frame_tau1_m1_cob = frame_tau1_m1.in_basis(curr_frame)
                    if tau1 == 0: # Forward diff
                        rot, rssd = R.align_vectors(frame_tau1_p1_cob.as_matrix().T, curr_frame.as_matrix().T)
                        d = np.linalg.norm(frame_tau1_p1_cob.x - curr_frame.x)
                    elif tau1 == 1: # Backward diff
                        rot, rssd = R.align_vectors(curr_frame.as_matrix().T, frame_tau1_m1_cob.as_matrix().T)
                        d = np.linalg.norm(curr_frame.x - frame_tau1_m1_cob.x)
                    else:
                        rot, rssd = R.align_vectors(frame_tau1_p1_cob.as_matrix().T, frame_tau1_m1_cob.as_matrix().T)
                        d = np.linalg.norm(frame_tau1_p1_cob.x - frame_tau1_m1_cob.x)
                    rotvec = rot.as_rotvec()
                    r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
                    features = features + [lat.value, lon.value, r.value/d]

    # Previous + 36 points * 6 features
    assert len(features) == 55*24 + 24*12 + 6*18 + 36*6

    # Next are onionskin fold points
    # Features are
    #   Location x change in the theta direction (length and direction)
    #   Curvatures in the theta direction, i.e., rotations per unit distance
    for theta in range(24):
        for level in [0.25,0.5,0.75]:
            curr_linear = crest_to_linear(theta)
            curr_frame = frames[levels.index(level)][curr_linear]

            # Location x change in the theta direction (length and direction)
            theta_plus = crest_to_linear(theta + 1)
            frame_plus = frames[levels.index(level)][theta_plus]
            theta_minus = crest_to_linear(theta - 1)
            frame_minus = frames[levels.index(level)][theta_minus]
            v = (frame_plus.x - frame_minus.x) / 2
            r, lat, lon = cartesian_to_spherical(v[0],v[1],v[2])
            features = features + [lat.value, lon.value, r.value] 
            
            # Curvatures in the theta direction, i.e., rotations per unit distance
            frame_plus_cob = frame_plus.in_basis(curr_frame)
            frame_minus_cob = frame_minus.in_basis(curr_frame)
            rot, rssd = R.align_vectors(frame_plus_cob.as_matrix().T, frame_minus_cob.as_matrix().T)
            d = np.linalg.norm(frame_plus_cob.x - frame_minus_cob.x)
            rotvec = rot.as_rotvec()
            r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
            features = features + [lat.value, lon.value, r.value/d]

    # Previous + 72 points * 6 features
    assert len(features) == 55*24 + 24*12 + 6*18 + 36*6 + 72*6

    # Next are interior onionskin points
    # Features are
    #   Location x change in the theta direction (length and direction)
    #   Location x change in the tau1 direction (length and direction)
    #   Curvatures in the theta direction, i.e., rotations per unit distance
    #   Curvatures in the tau1 direction, i.e., rotations per unit distance
    for theta in range(24):
        for tau1 in [0,0.5,1]:
            for tau2 in [0,1]:
                for level in [0.25,0.5,0.75]:
                    curr_linear = theta_tau_to_linear(theta,tau1,tau2)
                    curr_frame = frames[levels.index(level)][curr_linear]
                    if is_spine_extension(curr_linear) or is_repeated(curr_linear):
                        continue

                    # Location x change in the theta direction (length and direction)
                    theta_p1 = theta_tau_to_linear(theta+1,tau1,tau2)
                    frame_theta_p1 = frames[levels.index(level)][theta_p1]
                    theta_m1 = theta_tau_to_linear(theta-1,tau1,tau2)
                    frame_theta_m1 = frames[levels.index(level)][theta_m1]

                    if (is_spine_extension(theta_p1)): # Use backward difference
                        v = curr_frame.x - frame_theta_m1.x 
                    elif (is_spine_extension(theta_m1)):  # Use forward difference
                        v = frame_theta_p1.x - curr_frame.x 
                    else:
                        v = (frame_theta_p1.x - frame_theta_m1.x) / 2 
                    r, lat, lon = cartesian_to_spherical(v[0],v[1],v[2])
                    features = features + [lat.value, lon.value, r.value] 

                    # Location x change in the tau1 direction (length and direction)
                    tau1_p1 = theta_tau_to_linear(theta,tau1+0.5,tau2)
                    frame_tau1_p1 = frames[levels.index(level)][tau1_p1]

                    tau1_m1 = theta_tau_to_linear(theta,tau1-0.5,tau2)
                    frame_tau1_m1 = frames[levels.index(level)][tau1_m1]

                    if tau1 == 0:
                        v = frame_tau1_p1.x - curr_frame.x
                    elif tau1 == 1:
                        v = curr_frame.x - frame_tau1_m1.x
                    else:
                        v = (frame_tau1_p1.x - frame_tau1_m1.x) / 2
                    r, lat, lon = cartesian_to_spherical(v[0],v[1],v[2])
                    features = features + [lat.value, lon.value, r.value]

                    # Curvatures in the theta direction, i.e., rotations per unit distance
                    frame_theta_p1_cob = frame_theta_p1.in_basis(curr_frame)
                    frame_theta_m1_cob = frame_theta_m1.in_basis(curr_frame)
                    if (is_spine_extension(theta_p1)): # Use backward difference
                        rot, rssd = R.align_vectors(curr_frame.as_matrix().T, frame_theta_m1_cob.as_matrix().T)
                        d = np.linalg.norm(curr_frame.x - frame_theta_m1_cob.x)
                    elif (is_spine_extension(theta_m1)):  # Use forward difference
                        rot, rssd = R.align_vectors(frame_theta_p1_cob.as_matrix().T, curr_frame.as_matrix().T)
                        d = np.linalg.norm(frame_theta_p1_cob.x - curr_frame.x)
                    else:
                        rot, rssd = R.align_vectors(frame_theta_p1_cob.as_matrix().T, frame_theta_m1_cob.as_matrix().T)
                        d = np.linalg.norm(frame_theta_p1_cob.x - frame_theta_m1_cob.x)
                    rotvec = rot.as_rotvec()
                    r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
                    features = features + [lat.value, lon.value, r.value/d]

                    # Curvatures in the tau1 direction, i.e., rotations per unit distance
                    frame_tau1_p1_cob = frame_tau1_p1.in_basis(curr_frame)
                    frame_tau1_m1_cob = frame_tau1_m1.in_basis(curr_frame)
                    if tau1 == 0: # Forward diff
                        rot, rssd = R.align_vectors(frame_tau1_p1_cob.as_matrix().T, curr_frame.as_matrix().T)
                        d = np.linalg.norm(frame_tau1_p1_cob.x - curr_frame.x)
                    elif tau1 == 1: # Backward diff
                        rot, rssd = R.align_vectors(curr_frame.as_matrix().T, frame_tau1_m1_cob.as_matrix().T)
                        d = np.linalg.norm(curr_frame.x - frame_tau1_m1_cob.x)
                    else:
                        rot, rssd = R.align_vectors(frame_tau1_p1_cob.as_matrix().T, frame_tau1_m1_cob.as_matrix().T)
                        d = np.linalg.norm(frame_tau1_p1_cob.x - frame_tau1_m1_cob.x)
                    rotvec = rot.as_rotvec()
                    r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
                    features = features + [lat.value, lon.value, r.value/d]

    # Previous + 330 points * 12 features
    assert len(features) == 55*24 + 24*12 + 6*18 + 36*6 + 72*6 + 330*12

    # Next are boundary points
    # Features are
    #   Location x change in the theta direction (length and direction)
    #   Location x change in the tau1 direction (length and direction)
    #   Curvatures in the theta direction, i.e., rotations per unit distance
    #   Curvatures in the tau1 direction, i.e., rotations per unit distance
    for theta in range(24):
        for tau1 in [0,0.5,1]:
            for tau2 in [0,1]:
                for level in [1]:
                    curr_linear = theta_tau_to_linear(theta,tau1,tau2)
                    curr_frame = frames[levels.index(level)][curr_linear]
                    if is_spine_extension(curr_linear) or is_repeated(curr_linear):
                        continue

                    # Location x change in the theta direction (length and direction)
                    theta_p1 = theta_tau_to_linear(theta+1,tau1,tau2)
                    frame_theta_p1 = frames[levels.index(level)][theta_p1]
                    theta_m1 = theta_tau_to_linear(theta-1,tau1,tau2)
                    frame_theta_m1 = frames[levels.index(level)][theta_m1]

                    if (is_spine_extension(theta_p1)): # Use backward difference
                        v = curr_frame.x - frame_theta_m1.x 
                    elif (is_spine_extension(theta_m1)):  # Use forward difference
                        v = frame_theta_p1.x - curr_frame.x 
                    else:
                        v = (frame_theta_p1.x - frame_theta_m1.x) / 2 
                    r, lat, lon = cartesian_to_spherical(v[0],v[1],v[2])
                    features = features + [lat.value, lon.value, r.value] 

                    # Location x change in the tau1 direction (length and direction)
                    tau1_p1 = theta_tau_to_linear(theta,tau1+0.5,tau2)
                    frame_tau1_p1 = frames[levels.index(level)][tau1_p1]

                    tau1_m1 = theta_tau_to_linear(theta,tau1-0.5,tau2)
                    frame_tau1_m1 = frames[levels.index(level)][tau1_m1]

                    if tau1 == 0:
                        v = frame_tau1_p1.x - curr_frame.x
                    elif tau1 == 1:
                        v = curr_frame.x - frame_tau1_m1.x
                    else:
                        v = (frame_tau1_p1.x - frame_tau1_m1.x) / 2
                    r, lat, lon = cartesian_to_spherical(v[0],v[1],v[2])
                    features = features + [lat.value, lon.value, r.value]

                    # Curvatures in the theta direction, i.e., rotations per unit distance
                    frame_theta_p1_cob = frame_theta_p1.in_basis(curr_frame)
                    frame_theta_m1_cob = frame_theta_m1.in_basis(curr_frame)
                    if (is_spine_extension(theta_p1)): # Use backward difference
                        rot, rssd = R.align_vectors(curr_frame.as_matrix().T, frame_theta_m1_cob.as_matrix().T)
                        d = np.linalg.norm(curr_frame.x - frame_theta_m1_cob.x)
                    elif (is_spine_extension(theta_m1)):  # Use forward difference
                        rot, rssd = R.align_vectors(frame_theta_p1_cob.as_matrix().T, curr_frame.as_matrix().T)
                        d = np.linalg.norm(frame_theta_p1_cob.x - curr_frame.x)
                    else:
                        rot, rssd = R.align_vectors(frame_theta_p1_cob.as_matrix().T, frame_theta_m1_cob.as_matrix().T)
                        d = np.linalg.norm(frame_theta_p1_cob.x - frame_theta_m1_cob.x)
                    rotvec = rot.as_rotvec()
                    r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
                    features = features + [lat.value, lon.value, r.value/d]

                    # Curvatures in the tau1 direction, i.e., rotations per unit distance
                    frame_tau1_p1_cob = frame_tau1_p1.in_basis(curr_frame)
                    frame_tau1_m1_cob = frame_tau1_m1.in_basis(curr_frame)
                    if tau1 == 0: # Forward diff
                        rot, rssd = R.align_vectors(frame_tau1_p1_cob.as_matrix().T, curr_frame.as_matrix().T)
                        d = np.linalg.norm(frame_tau1_p1_cob.x - curr_frame.x)
                    elif tau1 == 1: # Backward diff
                        rot, rssd = R.align_vectors(curr_frame.as_matrix().T, frame_tau1_m1_cob.as_matrix().T)
                        d = np.linalg.norm(curr_frame.x - frame_tau1_m1_cob.x)
                    else:
                        rot, rssd = R.align_vectors(frame_tau1_p1_cob.as_matrix().T, frame_tau1_m1_cob.as_matrix().T)
                        d = np.linalg.norm(frame_tau1_p1_cob.x - frame_tau1_m1_cob.x)
                    rotvec = rot.as_rotvec()
                    r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
                    features = features + [lat.value, lon.value, r.value/d]
                    
    # Spine end/extensions boundary points
    for theta in range(24):
        for tau1 in [0,0.5,1]:
            for tau2 in [0,1]:
                for level in [1]:
                    curr_linear = theta_tau_to_linear(theta, tau1, tau2)
                    curr_frame = frames[levels.index(level)][curr_linear]
                    if not is_spine_extension(curr_linear):
                        continue

                    # Location x change in the theta direction (length and direction)
                    theta_p1 = theta_tau_to_linear(theta+1,tau1+0.5,tau2)
                    frame_theta_p1 = frames[levels.index(level)][theta_p1]
                    theta_m1 = theta_tau_to_linear(theta-1,tau1+0.5,tau2)
                    frame_theta_m1 = frames[levels.index(level)][theta_m1]

                    if (is_spine_extension(theta_p1)): # Use backward difference
                        v = curr_frame.x - frame_theta_m1.x 
                    elif (is_spine_extension(theta_m1)):  # Use forward difference
                        v = frame_theta_p1.x - curr_frame.x 
                    else:
                        v = (frame_theta_p1.x - frame_theta_m1.x) / 2 
                    r, lat, lon = cartesian_to_spherical(v[0],v[1],v[2])
                    features = features + [lat.value, lon.value, r.value] 

                    # Location x change in the tau1 direction (length and direction)
                    tau1_p1 = theta_tau_to_linear(theta,tau1+0.5,tau2)
                    frame_tau1_p1 = frames[levels.index(level)][tau1_p1]

                    tau1_m1 = theta_tau_to_linear(theta,tau1-0.5,tau2)
                    frame_tau1_m1 = frames[levels.index(level)][tau1_m1]

                    if tau1 == 0:
                        v = frame_tau1_p1.x - curr_frame.x
                    elif tau1 == 1:
                        v = curr_frame.x - frame_tau1_m1.x
                    else:
                        v = (frame_tau1_p1.x - frame_tau1_m1.x) / 2
                    r, lat, lon = cartesian_to_spherical(v[0],v[1],v[2])
                    features = features + [lat.value, lon.value, r.value]

                    # Curvatures in the theta direction, i.e., rotations per unit distance
                    frame_theta_p1_cob = frame_theta_p1.in_basis(curr_frame)
                    frame_theta_m1_cob = frame_theta_m1.in_basis(curr_frame)
                    if (is_spine_extension(theta_p1)): # Use backward difference
                        rot, rssd = R.align_vectors(curr_frame.as_matrix().T, frame_theta_m1_cob.as_matrix().T)
                        d = np.linalg.norm(curr_frame.x - frame_theta_m1_cob.x)
                    elif (is_spine_extension(theta_m1)):  # Use forward difference
                        rot, rssd = R.align_vectors(frame_theta_p1_cob.as_matrix().T, curr_frame.as_matrix().T)
                        d = np.linalg.norm(frame_theta_p1_cob.x - curr_frame.x)
                    else:
                        rot, rssd = R.align_vectors(frame_theta_p1_cob.as_matrix().T, frame_theta_m1_cob.as_matrix().T)
                        d = np.linalg.norm(frame_theta_p1_cob.x - frame_theta_m1_cob.x)
                    rotvec = rot.as_rotvec()
                    r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
                    features = features + [lat.value, lon.value, r.value/d]

                    # Curvatures in the tau1 direction, i.e., rotations per unit distance
                    frame_tau1_p1_cob = frame_tau1_p1.in_basis(curr_frame)
                    frame_tau1_m1_cob = frame_tau1_m1.in_basis(curr_frame)
                    if tau1 == 0: # Forward diff
                        rot, rssd = R.align_vectors(frame_tau1_p1_cob.as_matrix().T, curr_frame.as_matrix().T)
                        d = np.linalg.norm(frame_tau1_p1_cob.x - curr_frame.x)
                    elif tau1 == 1: # Backward diff
                        rot, rssd = R.align_vectors(curr_frame.as_matrix().T, frame_tau1_m1_cob.as_matrix().T)
                        d = np.linalg.norm(curr_frame.x - frame_tau1_m1_cob.x)
                    else:
                        rot, rssd = R.align_vectors(frame_tau1_p1_cob.as_matrix().T, frame_tau1_m1_cob.as_matrix().T)
                        d = np.linalg.norm(frame_tau1_p1_cob.x - frame_tau1_m1_cob.x)
                    rotvec = rot.as_rotvec()
                    r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
                    features = features + [lat.value, lon.value, r.value/d]

    # Previous + 122 points * 12 features
    assert len(features) == 55*24 + 24*12 + 6*18 + 36*6 + 72*6 + 330*12 + 122*12

    # Boundary fold points
    for theta in range(24):
        for level in [1]:
            curr_linear = crest_to_linear(theta)
            curr_frame = frames[levels.index(level)][curr_linear]

            # Location x change in the theta direction (length and direction)
            theta_p1 = crest_to_linear(theta+1)
            frame_theta_p1 = frames[levels.index(level)][theta_p1]
            theta_m1 = crest_to_linear(theta-1)
            frame_theta_m1 = frames[levels.index(level)][theta_m1]
            v = (frame_theta_p1.x - frame_theta_m1.x) / 2 
            r, lat, lon = cartesian_to_spherical(v[0],v[1],v[2])
            features = features + [lat.value, lon.value, r.value]

            # Location x change in the tau1 direction (length and direction)
            tau1_top = theta_tau_to_linear(theta,1,0)
            frame_tau1_top = frames[levels.index(level)][tau1_top]

            tau1_bot = theta_tau_to_linear(theta,1,1)
            frame_tau1_bot = frames[levels.index(level)][tau1_bot]
            v = (frame_tau1_top.x - frame_tau1_bot.x) / 2
            r, lat, lon = cartesian_to_spherical(v[0],v[1],v[2])
            features = features + [lat.value, lon.value, r.value]   

            # Curvatures in the theta direction, i.e., rotations per unit distance
            frame_theta_p1_cob = frame_theta_p1.in_basis(curr_frame)
            frame_theta_m1_cob = frame_theta_m1.in_basis(curr_frame)
            rot, rssd = R.align_vectors(frame_theta_p1_cob.as_matrix().T, frame_theta_m1_cob.as_matrix().T)
            d = np.linalg.norm(frame_theta_p1_cob.x - frame_theta_m1_cob.x)
            rotvec = rot.as_rotvec()
            r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
            features = features + [lat.value, lon.value, r.value/d]

            # Curvatures in the tau1 direction, i.e., rotations per unit distance
            frame_tau1_top_cob = frame_tau1_top.in_basis(curr_frame)
            frame_tau1_bot_cob = frame_tau1_bot.in_basis(curr_frame)
            rot, rssd = R.align_vectors(frame_tau1_top_cob.as_matrix().T, frame_tau1_bot_cob.as_matrix().T)
            d = np.linalg.norm(frame_tau1_top_cob.x - frame_tau1_bot_cob.x)
            rotvec = rot.as_rotvec()
            r, lat, lon = cartesian_to_spherical(rotvec[0],rotvec[1], rotvec[2])
            features = features + [lat.value, lon.value, r.value/d]

    # Previous + 24 points * 6 features
    assert len(features) == 55*24 + 24*12 + 6*18 + 36*6 + 72*6 + 330*12 + 122*12 + 24*12

    return features

  def euclideanize_all_blocks(sefl, feats):
    # All 3-tuples are in the form of [latitude,longitude,distance]
    # This standardization makes euclideanization easier
    euc_features = np.zeros(feats.shape)
    for i in range(int(feats.shape[1] / 3)):
        block = feats[:,3*i:3*i+3]
        unit_vecs = np.zeros(block.shape)
        lats = block[:,0]
        lons = block[:,1]
        rs = block[:,2]

        # Convert lat,lon to x,y,z unit vectors
        xs,ys,zs = spherical_to_cartesian(np.ones(rs.shape),lats,lons)
        unit_vecs[:,0] = xs
        unit_vecs[:,1] = ys
        unit_vecs[:,2] = zs

        # Compute PNS on unit vectors
        pns_model = PNS(unit_vecs.T, itype=9)
        pns_model.fit()
        resmat, PNS_coords = pns_model.output

        # Euclideanize distances
        r_norms = np.exp(np.log(rs) - np.mean(np.log(rs)))

        euc_features[:,3*i:3*i+2] = resmat.T
        euc_features[:,3*i+2] = r_norms.T

    return euc_features

  def get_srep_features(self, filename):
    # Read in the template
    template_reader = vtk.vtkPolyDataReader()
    template_reader.SetFileName(self.templatePath)
    template_reader.Update()

    # Read the data for the current s-rep
    with open(filename) as f:
      data = json.load(f)
      pd = self.json_to_spoke_format(data,template_reader.GetOutput())\

      # Convert to feature representation
      feats = self.build_features(pd)
      return feats
    


  def run(self, inputCSV):
    reader = csv.reader(open(inputCSV))
    all_features = []
    labels = []
    for json_file,group in reader:
      print(f"Processing {json_file}")
      all_features.append(self.get_srep_features(json_file))
      labels.append(int(group))

    euc_feats = self.euclideanize_all_blocks(np.array(all_features))
    labels = np.array(labels)
    
    # Fit HotellingT2 estimator to one class then test on the other
    sig_count = 0
    total = 0
    for i in range(0,euc_feats.shape[1],3):
      total += 1
      group0 = euc_feats[np.where( labels == 0 )[0],i:i+3]
      group1 = euc_feats[np.where( labels == 1 )[0],i:i+3]

      hotelling = HotellingT2()
      hotelling.fit(group0)

      t2_score = hotelling.score(group1)
      ucl = group0.shape[0] / (group0.shape[0] + 1) * hotelling.ucl_indep_

      print(f"Do the training set and the test set come from the same "
        f"distribution? {t2_score <= ucl}")
      if t2_score > ucl:
        sig_count += 1
      
    print(f'Found {sig_count} significant features out of {total} total')
    logging.info('Processing completed')


#
# SRepHypothesisTestingTest
#

class SRepHypothesisTestingTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_SRepHypothesisTesting1()

  def test_SRepHypothesisTesting1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    import tempfile
    import os

    logic = SRepHypothesisTestingLogic()

    with tempfile.TemporaryDirectory() as tempdir:
      tempdir = Path(tempdir)

      content1 = os.urandom(32)
      content2 = os.urandom(32)

      data = tempdir / 'data'
      data.mkdir()
      (data / 'file').write_bytes(content1)
      (data / 'sub').mkdir()
      (data / 'sub' / 'file').write_bytes(content2)

      output = tempdir / 'output'

      logic.run(data, output)

      self.assertTrue(output.exists())
      self.assertTrue((output / 'file').exists())
      self.assertEqual((output / 'file').read_bytes(), content1)

      self.assertTrue((output / 'sub').exists())
      self.assertTrue((output / 'file').exists())
      self.assertEqual((output / 'sub' / 'file').read_bytes(), content2)

    self.delayDisplay('Test passed')
