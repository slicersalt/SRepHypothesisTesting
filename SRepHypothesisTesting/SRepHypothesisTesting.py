import logging
import os
import shutil
from pathlib import Path

import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


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

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    try:
      self.logic.run(
      )
    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: {}".format(e))
      import traceback
      traceback.print_exc()


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

  def run(self):
    
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
