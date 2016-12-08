#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
=============================================
fMRI: BIDS data, FSL, ANTS, c3daffine
=============================================

A growing number of datasets are available on `OpenfMRI <http://openfmri.org>`_.
This script demonstrates how to use nipype to analyze a BIDS data set::

    python fmri_ants_bids.py /path/to/bids/dir
"""

from nipype import config
#config.enable_provenance()

import six

from glob import glob
import os
import re

from nipype import LooseVersion
from nipype import Workflow, Node, MapNode, JoinNode
from nipype.interfaces import (fsl, Function, ants, nipy)
import nipype.interfaces.freesurfer as fs
from nipype.interfaces.utility import Rename, Merge, IdentityInterface
from nipype.utils.filemanip import filename_to_list, list_to_filename
from nipype.interfaces.io import DataSink, FreeSurferSource
import nipype.algorithms.modelgen as model
import nipype.algorithms.rapidart as ra
from nipype.algorithms.confounds import TSNR
from nipype.interfaces.c3 import C3dAffineTool
from nipype.workflows.fmri.fsl import (create_modelfit_workflow,
                                       create_fixed_effects_flow)
import numpy as np

version = 0
if fsl.Info.version() and \
    LooseVersion(fsl.Info.version()) > LooseVersion('5.0.6'):
    version = 507

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

imports = ['import os',
           'import nibabel as nb',
           'import numpy as np',
           'import scipy as sp',
           'from nipype.utils.filemanip import filename_to_list, list_to_filename, split_filename',
           'from scipy.special import legendre'
           ]

def rename(in_files, suffix=None):
    """ Utility function to keep files unique """
    from nipype.utils.filemanip import (filename_to_list, split_filename,
                                        list_to_filename)
    out_files = []
    for idx, filename in enumerate(filename_to_list(in_files)):
        _, name, ext = split_filename(filename)
        if suffix is None:
            out_files.append(name + ('_%03d' % idx) + ext)
        else:
            out_files.append(name + suffix + ext)
    return list_to_filename(out_files)


def create_reg_workflow(name='registration'):
    """Create a FEAT preprocessing workflow together with freesurfer

    Parameters
    ----------
        name : name of workflow (default: 'registration')

    Inputs:

        inputspec.source_files : files (filename or list of filenames to register)
        inputspec.mean_image : reference image to use
        inputspec.anatomical_image : anatomical image to coregister to
        inputspec.target_image : registration target

    Outputs:

        outputspec.func2anat_transform : FLIRT transform
        outputspec.transformed_mean : mean image in target space

    Example
    -------
    """

    register = Workflow(name=name)

    inputnode = Node(IdentityInterface(fields=['source_files',
                                               'mean_image',
                                               'anatomical_image',
                                               'target_image',
                                               'config_file']),
                        name='inputspec')
    outputnode = Node(IdentityInterface(fields=['func2anat_transform',
                                                'anat2target_transform',
                                                'transformed_files',
                                                'transformed_mean',
                                                'anat2target',
                                                'mean2anat_mask']),
                         name='outputspec')

    """
    Estimate the tissue classes from the anatomical image. But use spm's segment
    as FSL appears to be breaking.
    """

    stripper = Node(fsl.BET(), name='stripper')
    register.connect(inputnode, 'anatomical_image', stripper, 'in_file')
    fast = Node(fsl.FAST(), name='fast')
    register.connect(stripper, 'out_file', fast, 'in_files')

    """
    Binarize the segmentation
    """

    binarize = Node(fsl.ImageMaths(op_string='-nan -thr 0.5 -bin'),
                       name='binarize')
    pickindex = lambda x, i: x[i]
    register.connect(fast, ('partial_volume_files', pickindex, 2),
                     binarize, 'in_file')

    """
    Calculate rigid transform from mean image to anatomical image
    """

    mean2anat = Node(fsl.FLIRT(), name='mean2anat')
    mean2anat.inputs.dof = 6
    register.connect(inputnode, 'mean_image', mean2anat, 'in_file')
    register.connect(stripper, 'out_file', mean2anat, 'reference')

    """
    Now use bbr cost function to improve the transform
    """

    mean2anatbbr = Node(fsl.FLIRT(), name='mean2anatbbr')
    mean2anatbbr.inputs.dof = 6
    mean2anatbbr.inputs.cost = 'bbr'
    mean2anatbbr.inputs.schedule = os.path.join(os.getenv('FSLDIR'),
                                                'etc/flirtsch/bbr.sch')
    register.connect(inputnode, 'mean_image', mean2anatbbr, 'in_file')
    register.connect(binarize, 'out_file', mean2anatbbr, 'wm_seg')
    register.connect(inputnode, 'anatomical_image', mean2anatbbr, 'reference')
    register.connect(mean2anat, 'out_matrix_file',
                     mean2anatbbr, 'in_matrix_file')

    """
    Create a mask of the median image coregistered to the anatomical image
    """
    
    mean2anat_mask = Node(fsl.BET(mask=True), name='mean2anat_mask')
    register.connect(mean2anatbbr, 'out_file', mean2anat_mask, 'in_file')

    """
    Convert the BBRegister transformation to ANTS ITK format
    """

    convert2itk = Node(C3dAffineTool(),
                          name='convert2itk')
    convert2itk.inputs.fsl2ras = True
    convert2itk.inputs.itk_transform = True
    register.connect(mean2anatbbr, 'out_matrix_file', convert2itk, 'transform_file')
    register.connect(inputnode, 'mean_image',convert2itk, 'source_file')
    register.connect(stripper, 'out_file', convert2itk, 'reference_file')

    """
    Compute registration between the subject's structural and MNI template
    This is currently set to perform a very quick registration. However, the
    registration can be made significantly more accurate for cortical
    structures by increasing the number of iterations
    All parameters are set using the example from:
    #https://github.com/stnava/ANTs/blob/master/Scripts/newAntsExample.sh
    """

    reg = Node(ants.Registration(), name='antsRegister')
    reg.inputs.output_transform_prefix = "output_"
    reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.2, 3.0, 0.0)]
    reg.inputs.number_of_iterations = [[10000, 11110, 11110]] * 2 + [[100, 30, 20]]
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = True
    reg.inputs.initial_moving_transform_com = True
    reg.inputs.metric = ['Mattes'] * 2 + [['Mattes', 'CC']]
    reg.inputs.metric_weight = [1] * 2 + [[0.5, 0.5]]
    reg.inputs.radius_or_number_of_bins = [32] * 2 + [[32, 4]]
    reg.inputs.sampling_strategy = ['Regular'] * 2 + [[None, None]]
    reg.inputs.sampling_percentage = [0.3] * 2 + [[None, None]]
    reg.inputs.convergence_threshold = [1.e-8] * 2 + [-0.01]
    reg.inputs.convergence_window_size = [20] * 2 + [5]
    reg.inputs.smoothing_sigmas = [[4, 2, 1]] * 2 + [[1, 0.5, 0]]
    reg.inputs.sigma_units = ['vox'] * 3
    reg.inputs.shrink_factors = [[3, 2, 1]]*2 + [[4, 2, 1]]
    reg.inputs.use_estimate_learning_rate_once = [True] * 3
    reg.inputs.use_histogram_matching = [False] * 2 + [True]
    reg.inputs.winsorize_lower_quantile = 0.005
    reg.inputs.winsorize_upper_quantile = 0.995
    reg.inputs.args = '--float'
    reg.inputs.output_warped_image = 'output_warped_image.nii.gz'
    reg.inputs.num_threads = 4
    reg.plugin_args = {'qsub_args': '-pe orte 4',
                       'sbatch_args': '--mem=6G -c 4'}
    register.connect(stripper, 'out_file', reg, 'moving_image')
    register.connect(inputnode,'target_image', reg,'fixed_image')


    """
    Concatenate the affine and ants transforms into a list
    """

    pickfirst = lambda x: x[0]

    merge = Node(Merge(2), iterfield=['in2'], name='mergexfm')
    register.connect(convert2itk, 'itk_transform', merge, 'in2')
    register.connect(reg, 'composite_transform', merge, 'in1')


    """
    Transform the mean image. First to anatomical and then to target
    """

    warpmean = Node(ants.ApplyTransforms(),
                       name='warpmean')
    warpmean.inputs.input_image_type = 0
    warpmean.inputs.interpolation = 'Linear'
    warpmean.inputs.invert_transform_flags = [False, False]
    warpmean.inputs.terminal_output = 'file'

    register.connect(inputnode,'target_image', warpmean,'reference_image')
    register.connect(inputnode, 'mean_image', warpmean, 'input_image')
    register.connect(merge, 'out', warpmean, 'transforms')

    """
    Transform the remaining images. First to anatomical and then to target
    """

    warpall = MapNode(ants.ApplyTransforms(),
                         iterfield=['input_image'],
                         name='warpall')
    warpall.inputs.input_image_type = 0
    warpall.inputs.interpolation = 'Linear'
    warpall.inputs.invert_transform_flags = [False, False]
    warpall.inputs.terminal_output = 'file'

    register.connect(inputnode,'target_image',warpall,'reference_image')
    register.connect(inputnode,'source_files', warpall, 'input_image')
    register.connect(merge, 'out', warpall, 'transforms')


    """
    Assign all the output files
    """

    register.connect(reg, 'warped_image', outputnode, 'anat2target')
    register.connect(warpmean, 'output_image', outputnode, 'transformed_mean')
    register.connect(warpall, 'output_image', outputnode, 'transformed_files')
    register.connect(mean2anatbbr, 'out_matrix_file',
                     outputnode, 'func2anat_transform')
    register.connect(mean2anat_mask, 'mask_file',
                     outputnode, 'mean2anat_mask')
    register.connect(reg, 'composite_transform',
                     outputnode, 'anat2target_transform')

    return register

def get_aparc_aseg(files):
    """Return the aparc+aseg.mgz file"""
    for name in files:
        if 'aparc+aseg.mgz' in name:
            return name
    raise ValueError('aparc+aseg.mgz not found')

def create_fs_reg_workflow(name='registration'):
    """Create a FEAT preprocessing workflow together with freesurfer
    Parameters
    ----------
        name : name of workflow (default: 'registration')
    Inputs::
        inputspec.source_files : files (filename or list of filenames to register)
        inputspec.mean_image : reference image to use
        inputspec.subject_id : subject id
        inputspec.subjects_dir : path to freesurfer recon
        inputspec.target_image : registration target
    Outputs::
        outputspec.func2anat_transform : FLIRT transform
        outputspec.anat2target_transform : FLIRT+FNIRT transform
        outputspec.transformed_mean : mean image in target space
        outputspec.out_reg_file : 
        outputspec.segmentation_files : 
        outputspec.aparc : 
        outputspec.min_cost_file : 
    """

    register = Workflow(name=name)

    inputnode = Node(IdentityInterface(fields=['source_files',
                                               'mean_image',
                                               'subject_id',
                                               'subjects_dir',
                                               'target_image']),
                     name='inputspec')

    outputnode = Node(IdentityInterface(fields=['func2anat_transform',
                                                'out_reg_file',
                                                'anat2target_transform',
                                                'transforms',
                                                'transformed_files',
                                                'transformed_mean',
                                                'segmentation_files',
                                                'anat2target',
                                                'aparc',
                                                'min_cost_file']),
                      name='outputspec')

    # Get the subject's freesurfer source directory
    fssource = Node(FreeSurferSource(),
                    name='fssource')
    fssource.run_without_submitting = True
    register.connect(inputnode, 'subject_id', fssource, 'subject_id')
    register.connect(inputnode, 'subjects_dir', fssource, 'subjects_dir')

    convert = Node(fs.MRIConvert(out_type='nii'),
                   name="convert")
    register.connect(fssource, 'T1', convert, 'in_file')

    # Coregister the median to the surface
    bbregister = Node(fs.BBRegister(),
                      name='bbregister')
    bbregister.inputs.init = 'fsl'
    bbregister.inputs.contrast_type = 't2'
    bbregister.inputs.out_fsl_file = True
    bbregister.inputs.epi_mask = True
    register.connect(inputnode, 'subject_id', bbregister, 'subject_id')
    register.connect(inputnode, 'mean_image', bbregister, 'source_file')
    register.connect(inputnode, 'subjects_dir', bbregister, 'subjects_dir')

    """
    Estimate the tissue classes from the anatomical image. But use aparc+aseg's brain mask
    """
    binarize = Node(fs.Binarize(min=0.5, out_type="nii.gz", dilate=1), name="binarize_aparc")
    register.connect(fssource, ("aparc_aseg", get_aparc_aseg), binarize, "in_file")
    stripper = Node(fsl.ApplyMask(), name='stripper')
    register.connect(binarize, "binary_file", stripper, "mask_file")
    register.connect(convert, 'out_file', stripper, 'in_file')

    fast = Node(fsl.FAST(), name='fast')
    register.connect(stripper, 'out_file', fast, 'in_files')

    """
    Binarize the segmentation
    """
    binarize = MapNode(fsl.ImageMaths(op_string='-nan -thr 0.9 -ero -bin'),
                       iterfield=['in_file'],
                       name='binarize')
    register.connect(fast, 'partial_volume_files', binarize, 'in_file')

    """
    Apply inverse transform to take segmentations to functional space
    """
    applyxfm = MapNode(fs.ApplyVolTransform(inverse=True,
                                            interp='nearest'),
                       iterfield=['target_file'],
                       name='inverse_transform')
    register.connect(inputnode, 'subjects_dir', applyxfm, 'subjects_dir')
    register.connect(bbregister, 'out_reg_file', applyxfm, 'reg_file')
    register.connect(binarize, 'out_file', applyxfm, 'target_file')
    register.connect(inputnode, 'mean_image', applyxfm, 'source_file')

    """
    Apply inverse transform to aparc file
    """
    aparcxfm = Node(fs.ApplyVolTransform(inverse=True,
                                         interp='nearest'),
                    name='aparc_inverse_transform')
    register.connect(inputnode, 'subjects_dir', aparcxfm, 'subjects_dir')
    register.connect(bbregister, 'out_reg_file', aparcxfm, 'reg_file')
    register.connect(fssource, ('aparc_aseg', get_aparc_aseg),
                     aparcxfm, 'target_file')
    register.connect(inputnode, 'mean_image', aparcxfm, 'source_file')

    """
    Convert the BBRegister transformation to ANTS ITK format
    """
    convert2itk = Node(C3dAffineTool(), name='convert2itk')
    convert2itk.inputs.fsl2ras = True
    convert2itk.inputs.itk_transform = True
    register.connect(bbregister, 'out_fsl_file', convert2itk, 'transform_file')
    register.connect(inputnode, 'mean_image', convert2itk, 'source_file')
    register.connect(stripper, 'out_file', convert2itk, 'reference_file')

    """
    Compute registration between the subject's structural and MNI template
    This is currently set to perform a very quick registration. However, the
    registration can be made significantly more accurate for cortical
    structures by increasing the number of iterations
    All parameters are set using the example from:
    #https://github.com/stnava/ANTs/blob/master/Scripts/newAntsExample.sh
    """
    reg = Node(ants.Registration(), name='antsRegister')
    reg.inputs.output_transform_prefix = "output_"
    reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.2, 3.0, 0.0)]
    reg.inputs.number_of_iterations = [[10000, 11110, 11110]] * 2 + [[100, 30, 20]]
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = True
    reg.inputs.initial_moving_transform_com = True
    reg.inputs.metric = ['Mattes'] * 2 + [['Mattes', 'CC']]
    reg.inputs.metric_weight = [1] * 2 + [[0.5, 0.5]]
    reg.inputs.radius_or_number_of_bins = [32] * 2 + [[32, 4]]
    reg.inputs.sampling_strategy = ['Regular'] * 2 + [[None, None]]
    reg.inputs.sampling_percentage = [0.3] * 2 + [[None, None]]
    reg.inputs.convergence_threshold = [1.e-8] * 2 + [-0.01]
    reg.inputs.convergence_window_size = [20] * 2 + [5]
    reg.inputs.smoothing_sigmas = [[4, 2, 1]] * 2 + [[1, 0.5, 0]]
    reg.inputs.sigma_units = ['vox'] * 3
    reg.inputs.shrink_factors = [[3, 2, 1]] * 2 + [[4, 2, 1]]
    reg.inputs.use_estimate_learning_rate_once = [True] * 3
    reg.inputs.use_histogram_matching = [False] * 2 + [True]
    reg.inputs.winsorize_lower_quantile = 0.005
    reg.inputs.winsorize_upper_quantile = 0.995
    reg.inputs.float = True
    reg.inputs.output_warped_image = 'output_warped_image.nii.gz'
    reg.inputs.num_threads = 4
    reg.plugin_args = {'sbatch_args': '-c%d' % 4}
    register.connect(stripper, 'out_file', reg, 'moving_image')
    register.connect(inputnode, 'target_image', reg, 'fixed_image')

    """
    Concatenate the affine and ants transforms into a list
    """
    merge = Node(Merge(2), iterfield=['in2'], name='mergexfm')
    register.connect(convert2itk, 'itk_transform', merge, 'in2')
    register.connect(reg, 'composite_transform', merge, 'in1')

    """
    Transform the mean image. First to anatomical and then to target
    """
    warpmean = Node(ants.ApplyTransforms(), name='warpmean')
    warpmean.inputs.input_image_type = 3
    warpmean.inputs.interpolation = 'Linear'
    warpmean.inputs.invert_transform_flags = [False, False]
    warpmean.inputs.terminal_output = 'file'
    warpmean.inputs.args = '--float'
    warpmean.inputs.num_threads = 4
    warpmean.plugin_args = {'sbatch_args': '-c%d' % 4}

    register.connect(inputnode, 'target_image', warpmean, 'reference_image')
    register.connect(inputnode, 'mean_image', warpmean, 'input_image')
    register.connect(merge, 'out', warpmean, 'transforms')
    
    """
    Transform the remaining images. First to anatomical and then to target
    """
    warpall = MapNode(ants.ApplyTransforms(),
                         iterfield=['input_image'],
                         name='warpall')
    warpall.inputs.input_image_type = 0
    warpall.inputs.interpolation = 'Linear'
    warpall.inputs.invert_transform_flags = [False, False]
    warpall.inputs.terminal_output = 'file'
    warpall.inputs.args = '--float'

    register.connect(inputnode, 'target_image', warpall, 'reference_image')
    register.connect(inputnode, 'source_files', warpall, 'input_image')
    register.connect(merge, 'out', warpall, 'transforms')
    register.connect(warpall, 'output_image', outputnode, 'transformed_files')
    """
    Assign all the output files
    """
    register.connect(reg, 'warped_image', outputnode, 'anat2target')
    register.connect(warpmean, 'output_image', outputnode, 'transformed_mean')
    register.connect(applyxfm, 'transformed_file',
                     outputnode, 'segmentation_files')
    register.connect(aparcxfm, 'transformed_file',
                     outputnode, 'aparc')
    register.connect(bbregister, 'out_fsl_file',
                     outputnode, 'func2anat_transform')
    register.connect(bbregister, 'out_reg_file',
                     outputnode, 'out_reg_file')
    register.connect(reg, 'composite_transform',
                     outputnode, 'anat2target_transform')
    register.connect(merge, 'out', outputnode, 'transforms')
    register.connect(bbregister, 'min_cost_file',
                     outputnode, 'min_cost_file')
    return register

def create_topup_workflow(num_slices, readout, 
                          readout_topup, name='topup'):
    """Create a geometric distortion correction workflow using TOPUP 
    Parameters
    ----------
    name : name of workflow (default: 'topup')
    Inputs::
        inputspec.realigned_files : realigned bold time series files
        inputspec.ref_file : reference image to register TOPUP images to realigned files
        inputspec.topup_AP : merged TOPUP images in AP phase-encoding direction
        inputspec.topup_PA : merged TOPUP images in PA phase-encoding direction
        inputspec.phase_encoding : PE direction of run
    Outputs::
        outputspec.topup_encoding_file : acquisition parameter text file for TOPUP files
        outputspec.rest_encoding_file : acquisition parameter text file for rest file
        outputspec.topup_fieldcoef : spline coefficients encoding the off-resonance field
        outputspec.topup_movpar : TOPUP movement parameters output file 
        outputspec.topup_corrected : corrected TOPUP file
        outputspec.applytopup_corrected : corrected resting state time series 
    """

    topup = Workflow(name=name)

    inputnode = Node(IdentityInterface(fields=['realigned_files',
                                              'ref_file',
                                              'topup_AP',
                                              'topup_PA',
                                              'phase_encoding']),
                     name='inputspec')

    outputnode = Node(IdentityInterface(fields=['topup_encoding_file',
                                                'rest_encoding_file',
                                                'topup_fieldcoef',
                                                'topup_movpar',
                                                'topup_corrected',
                                                'applytopup_corrected']),
                      name='outputspec')

    topup2median = Node(fsl.FLIRT(out_file='orig2median.nii.gz', 
                                  output_type='NIFTI_GZ', interp='spline'),
                        name='orig2median')
    topup2median.inputs.dof = 6
    topup2median.inputs.out_matrix_file = 'orig2median'

    applyxfm = Node(fsl.ApplyXfm(out_file='opp2median.nii.gz', apply_xfm=True, 
                                 interp='spline', output_type='NIFTI_GZ'),
                    name='applyxfm')        
    topup.connect(topup2median, 'out_matrix_file', applyxfm, 'in_matrix_file')

    make_topup_list = Node(Merge(2), name='make_topup_list')
    topup.connect(topup2median, 'out_file', make_topup_list, 'in1')
    topup.connect(applyxfm, 'out_file', make_topup_list, 'in2')

    merge_topup = Node(fsl.Merge(dimension='t', output_type='NIFTI_GZ'), 
                        name='merge_topup')
    topup.connect(make_topup_list, 'out', merge_topup, 'in_files')

    def write_encoding_file(readout, fname, pe=None):
        """ Write topup encoding file """
        import os
        if pe == 'j':
            direction = 1
        elif (pe == 'j-') or not pe:
            direction = -1
        filename = os.path.join(os.getcwd(), 'acq_param_%s.txt' % fname)
        with open(filename, 'w') as f:  
            f.writelines(['0 %d 0 %s\n' % (direction, readout),    
                          '0 %d 0 %s\n' % (direction * -1, readout)])
        return filename

    file_writer_topup = Node(Function(input_names=['readout', 'fname',
                                                   'pe'],
                                output_names=['encoding_file'],
                                function=write_encoding_file),
                       name='file_writer_topup')
    file_writer_topup.inputs.readout = readout_topup
    file_writer_topup.inputs.fname = 'topup'
    topup.connect(inputnode, 'phase_encoding', file_writer_topup, 'pe')

    run_topup = Node(fsl.TOPUP(out_corrected='b0correct.nii.gz', numprec='float', 
                        config='b02b0.cnf', output_type='NIFTI_GZ'), 
                    name='run_topup')
    topup.connect(file_writer_topup, 'encoding_file', run_topup, 'encoding_file')

    applytopup = Node(fsl.ApplyTOPUP(output_type='NIFTI_GZ'), name='applytopup')
    applytopup.inputs.in_index = [1]
    applytopup.inputs.method = 'jac'   

    file_writer_ts = file_writer_topup.clone(name='file_writer_ts')
    file_writer_ts.inputs.readout = readout
    file_writer_ts.inputs.fname = 'rest_ts'
    topup.connect(inputnode, 'phase_encoding', file_writer_ts, 'pe')

    topup.connect(merge_topup, 'merged_file', run_topup, 'in_file')
    topup.connect(file_writer_ts, 'encoding_file', applytopup, 'encoding_file')
    topup.connect(run_topup, 'out_fieldcoef', applytopup, 'in_topup_fieldcoef')
    topup.connect(run_topup, 'out_movpar', applytopup, 'in_topup_movpar')

    if num_slices % 2 != 0:    

        rm_slice_ts = Node(fsl.ExtractROI(), name='rm_slice_ts')        
        rm_slice_ts.inputs.crop_list = [(0,-1), (0,-1), (0, num_slices-1), (0,-1)]

        rm_slice_ref = Node(fsl.ExtractROI(), name='rm_slice_ref') 
        rm_slice_ref.inputs.crop_list = [(0,-1),(0,-1),(0, num_slices-1),(0,1)]

        extract_main = rm_slice_ref.clone(name='extract_main') 

        extract_opp = extract_main.clone(name='extract_opp')  

        topup.connect([(inputnode, rm_slice_ts, [('realigned_files', 'in_file')]),
                     (rm_slice_ts, applytopup, [('roi_file', 'in_files')]),
                     (inputnode, rm_slice_ref, [('ref_file', 'in_file')]),
                     (rm_slice_ref, topup2median, [('roi_file', 'reference')]),
                     (rm_slice_ref, applyxfm, [('roi_file', 'reference')])
                     ])

    else:
        topup.connect([(inputnode, applytopup, [('realigned_files', 'in_files')]),
                     (inputnode, topup2median, [('ref_file', 'reference')]),
                     (inputnode, applyxfm, [('ref_file', 'reference')])])

        extract_main = Node(fsl.ExtractROI(), name='extract_main') 
        extract_main.inputs.crop_list = [(0,-1), (0,-1), (0,-1), (0,1)]

        extract_opp = extract_main.clone(name='extract_opp')
    
    def check_topup_dir(tAP, tPA, pe):
        """ Helper Function to return orig and opposite fieldmaps """
        if pe == 'j':
            return tPA, tAP
        elif pe == 'j-':
            return tAP, tPA

    check_dirs = Node(Function(input_names=['tAP', 'tPA', 'pe'],
                               output_names=['main', 'opp'],
                               function=check_topup_dir),
                      name='organize_dirs')

    topup.connect([(inputnode, check_dirs, [('topup_AP', 'tAP'),
                                            ('topup_PA', 'tPA'),
                                            ('phase_encoding', 'pe')]),
                   (check_dirs, extract_main, [('main', 'in_file')]),
                   (check_dirs, extract_opp, [('opp', 'in_file')])])

    topup.connect(extract_main, 'roi_file', topup2median, 'in_file')
    topup.connect(extract_opp, 'roi_file', applyxfm, 'in_file')

    topup.connect(file_writer_topup, 'encoding_file', outputnode, 'topup_encoding_file')
    topup.connect(file_writer_ts, 'encoding_file', outputnode, 'rest_encoding_file')
    topup.connect(run_topup, 'out_fieldcoef', outputnode, 'topup_fieldcoef')
    topup.connect(run_topup, 'out_movpar', outputnode, 'topup_movpar')
    topup.connect(run_topup, 'out_corrected', outputnode, 'topup_corrected')
    topup.connect(applytopup, 'out_corrected', outputnode, 'applytopup_corrected')
    return topup

def get_subjectinfo(subject_id, base_dir, taskname, model_id, session_id=None):
    """Get info for a given subject
    Parameters
    ----------
    subject_id : string
        Subject identifier (e.g., sub001)
    base_dir : string
        Path to base directory of the dataset
    task : str
        Which task to process (task-%s)
    model_id : int
        Which model to process
    Returns
    -------
    run_ids : list of ints
        Run numbers
    conds : list of str
        Condition names
    TR : float
        Repetition time
    """
    condition_info = []
    cond_file = os.path.join(base_dir, 'code', 'model', 'model%03d' % model_id,
                                 'condition_key.txt')
    task = 'task-{}'.format(taskname)
    with open(cond_file, 'rt') as fp:
        for line in fp:
            info = line.strip().split()
            condition_info.append([info[0], info[1], ' '.join(info[2:])])
    
    if len(condition_info) == 0:
        raise ValueError('No condition info found in %s' % cond_file)
        
    taskinfo = np.array(condition_info)
    n_tasks = []
    for x in taskinfo[:, 0]:
        if x not in n_tasks:
            n_tasks.append(x)
    conds = []
    run_ids = []
    if task not in n_tasks:
        raise ValueError('Task id %s does not exist' % task)
    for idx,taskname in enumerate(n_tasks):
        taskidx = np.where(taskinfo[:, 0] == '%s'%(taskname))
        conds.append([condition.replace(' ', '_') for condition
                      in taskinfo[taskidx[0], 2]])
        if session_id:
            files = sorted(glob(os.path.join(base_dir,
                                             subject_id,
                                             session_id,
                                             'func',
                                             '*%s*.nii.gz'%(task))))
        else:
            files = sorted(glob(os.path.join(base_dir,
                                             subject_id,
                                             'func',
                                             '*%s*.nii.gz'%(task))))
            
        try:
            runs = [int(re.search('(?<=run-)\d+',os.path.basename(val)).group(0)) for val in files]
        except AttributeError:
            runs = [int(re.search('(?<=run)\d+',os.path.basename(val)).group(0)) for val in files]
        run_ids.insert(idx, runs)
    if session_id:
        json_info = glob(os.path.join(base_dir, subject_id, session_id, 
                                      'func','*%s*.json'%(task)))[0]
    else:    
        json_info = glob(os.path.join(base_dir, subject_id, 'func',
                                     '*%s*.json'%(task)))[0]
    if os.path.exists(json_info):
        import json
        with open(json_info, 'rt') as fp:
            data = json.load(fp)
            TR = data['RepetitionTime']
            slice_times = data['SliceTiming']
    else:
        task_scan_key = os.path.join(base_dir, 'code', 'scan_key.txt')
        if os.path.exists(task_scan_key):
            TR = np.genfromtxt(task_scan_key)[1]
        else:
            TR = np.genfromtxt(os.path.join(base_dir, 'scan_key.txt'))[1]
    return run_ids[0], conds[n_tasks.index(task)], TR, slice_times

def get_topup_info(layout, bold_files):
    """
    Get info for topup correction
    """
    # same across runs so just get once
    if 'epi' in layout.get_fieldmap(bold_files[0])['type']:
        # parse each bold for AP/PA fmap
        for bold in bold_files:
            fmap = layout.get_fieldmap(bold)['epi'][0]
            if 'dir-AP' in fmap:
                topup_AP = fmap
            elif 'dir-PA' in fmap:
                topup_PA = fmap
            readout_topup = layout.get_metadata(fmap)['TotalReadoutTime']
    # metainfo (all the same)
    orig_info = layout.get_metadata(bold_files[0])
    num_slices, readout = orig_info['dcmmeta_shape'][3], \
    (orig_info['dcmmeta_shape'][0] - 1) * orig_info['EffectiveEchoSpacing']
    # keep track of directions
    pe_key = [layout.get_metadata(bold)['PhaseEncodingDirection'] for bold in bold_files]
    return num_slices, pe_key, readout, topup_AP, topup_PA, readout_topup

def create_workflow(bids_dir, args, fs_dir, derivatives, workdir, outdir):
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    subjs_to_analyze = []
    if args.subject:
        subjs_to_analyze = ['sub-{}'.format(val) for val in args.subject]
    else:
        subj_dirs = sorted(glob(os.path.join(bids_dir, 'sub-*')))
        subjs_to_analyze = [subj_dir.split(os.sep)[-1] for subj_dir in subj_dirs]

    old_model_dir = os.path.join(os.path.join(bids_dir, 'code', 'model', 
                                              'model{:0>3d}'.format(args.model)))

    contrast_file = os.path.join(old_model_dir, 'task_contrasts.txt')

    task_id = args.task

    # the master workflow, with subject specific inside
    meta_wf = Workflow('meta_level')

    print(subjs_to_analyze)
    for subj_label in subjs_to_analyze: #replacing infosource
        from bids.grabbids import BIDSLayout
        layout = BIDSLayout(bids_dir)
        
        if task_id not in layout.get_tasks():
            print('task-{} is not found in your dataset'.format(task_id))
            return

        run_id, conds, TR, slice_times = get_subjectinfo(subj_label, bids_dir, 
                                                         task_id, args.model)
        # replacing datasource
        bold_files = sorted([f.filename for f in \
                      layout.get(subject = subj_label.replace('sub-',''),
                      type='bold', task=task_id, extensions=['nii.gz', 'nii'])])

        anat = [f.filename for f in \
                layout.get(subject = subj_label.replace('sub-',''),
                type='T1w', extensions=['nii.gz', 'nii'])][0]

        #use events.tsv eventually
        behav = [x for x in glob(
                 os.path.join(old_model_dir, 'test_onsets', subj_label,
                         'task-{}*'.format(task_id), 'cond*.txt'))]

        name = '{sub}_task-{task}'.format(sub=subj_label, task=task_id)

        # until slice timing is fixed, don't use
        kwargs = dict(bold_files=bold_files,
                      anat=anat,
                      target_file=args.target_file,
                      subject_id=subj_label,
                      task_id=task_id,
                      model_id=args.model,
                      TR=TR,
                      slice_times=None,
                      behav=behav,
                      fs_dir=fs_dir,
                      conds=conds,
                      run_id=run_id,
                      highpass_freq=args.hpfilter,
                      lowpass_freq=args.lpfilter,
                      fwhm=args.fwhm,
                      contrast=contrast_file,
                      use_derivatives=derivatives,
                      outdir=os.path.join(outdir, subj_label),
                      name=name)
        # add flag for topup
        if args.topup:
            num_slices, pe_key, readout, \
            topup_AP, topup_PA, readout_topup = get_topup_info(layout, #run_id,
                                                               bold_files)

            topup_kwargs = dict(num_slices=num_slices,
                                pe_key=pe_key,
                                readout=readout,
                                topup_AP=topup_AP,
                                topup_PA=topup_PA,
                                readout_topup=readout_topup)

            kwargs = dict(kwargs.items() + topup_kwargs.items())
                      
        wf = analyze_bids_dataset(**kwargs)
        meta_wf.add_nodes([wf])
    return meta_wf

def analyze_bids_dataset(bold_files, 
                         anat, 
                         subject_id, 
                         task_id, 
                         model_id, 
                         TR,
                         behav,
                         slice_times=None,
                         target_file=None, 
                         fs_dir=None, 
                         conds=None,
                         run_id=None,
                         highpass_freq=0.01,
                         lowpass_freq=0.1, 
                         fwhm=6., 
                         contrast=None,
                         use_derivatives=True,
                         num_slices=None,
                         pe_key=None,
                         readout=None,
                         topup_AP=None,
                         topup_PA=None,
                         readout_topup=None, 
                         outdir=None, 
                         name='tfmri'):
    
    # Initialize subject workflow and import others
    wf = Workflow(name=name)
    modelfit = create_modelfit_workflow()
    modelfit.inputs.inputspec.interscan_interval = TR
    fixed_fx = create_fixed_effects_flow()

    # Start of bold analysis
    if pe_key:
        infosource = Node(IdentityInterface(fields=['bold', 'pe']),
                      name='infosource')
        infosource.iterables = [('bold', bold_files), ('pe', pe_key)]
        infosource.synchronize = True
    else:
        infosource = Node(IdentityInterface(fields=['bold']),
                      name='infosource')
        infosource.iterables = ('bold', bold_files)

    # if no slice_times, SpatialRealign algorithm used
    realign_run = Node(nipy.SpaceTimeRealigner(), 
                       name='realign_per_run')
    if slice_times:
        realign_run.inputs.slice_times = slice_times
        realign_run.inputs.slice_info = 2
        realign_run.inputs.tr = TR
    realign_run.plugin_args = {'sbatch_args': '-c 4'}
    wf.connect(infosource, 'bold', realign_run, 'in_file')

    # Comute TSNR on realigned data regressing polynomials upto order 2
    tsnr = Node(TSNR(regress_poly=2), name='tsnr')
    tsnr.plugin_args = {'qsub_args': '-pe orte 4',
                       'sbatch_args': '--mem=16G -c 4'}

    def median(in_files):
        """Computes an average of the median of each realigned timeseries
        Parameters
        ----------
        in_files: one or more realigned Nifti 4D time series

        Returns
        -------
        out_file: a 3D Nifti file
        """
        average = None
        for idx, filename in enumerate(filename_to_list(in_files)):
            img = nb.load(filename)
            data = np.median(img.get_data(), axis=3)
            if average is None:
                average = data
            else:
                average = average + data
        median_img = nb.Nifti1Image(average/float(idx + 1),
                                    img.get_affine(), img.get_header())
        filename = os.path.join(os.getcwd(), 'median.nii.gz')
        median_img.to_filename(filename)
        return filename

    # Compute the median image across runs
    calc_median = Node(Function(input_names=['in_files'],
                                output_names=['median_file'],
                                function=median, imports=imports),
                          name='median')
    
    # regardless of topup makes workflow easier to connect
    recalc_median = calc_median.clone(name='recalc_median')

    if topup_AP:
        topup = create_topup_workflow(num_slices, readout, 
                                      readout_topup, name='topup')
        topup.inputs.inputspec.topup_AP = topup_AP
        topup.inputs.inputspec.topup_PA = topup_PA
        wf.connect(infosource, 'pe', topup, 'inputspec.phase_encoding')
        wf.connect(realign_run, 'out_file', topup, 'inputspec.realigned_files')
        wf.connect(realign_run, 'out_file', calc_median, 'in_files')
        wf.connect(calc_median, 'median_file', topup, 'inputspec.ref_file')

        def topup_combiner(topup_corrected_bold, realign_movpar, topup_movpar):
            """ joiner after topup correction """
            return topup_corrected_bold, realign_movpar, topup_movpar

        joiner = JoinNode(Function(input_names=['topup_corrected_bold',
                                                'realign_movpar',
                                                'topup_movpar'],
                                   output_names=['corrected_bolds',
                                                 'nipy_realign_pars',
                                                 'topup_movpars'],
                                   function=topup_combiner),
                          joinsource='infosource',
                          joinfield=['topup_corrected_bold',
                                     'realign_movpar',
                                     'topup_movpar'],
                          name='topup_joiner')
        # basically a datasink for each run
        wf.connect(topup, 'outputspec.applytopup_corrected', joiner, 'topup_corrected_bold')
        wf.connect(realign_run, 'par_file', joiner, 'realign_movpar')
        wf.connect(topup, 'outputspec.topup_movpar', joiner, 'topup_movpar')
    else:
        
        def run_combiner(bold_file, realign_movpar):
            """ joiner for non-topup """
            return bold_file, realign_movpar

        joiner = JoinNode(Function(input_names=['bold_file',
                                                'realign_movpar'],
                                   output_names=['corrected_bolds',
                                                 'nipy_realign_pars'],
                                   function=run_combiner),
                          joinsource='infosource',
                          joinfield=['bold_file', 'realign_movpar'],
                          name='run_joiner')
        wf.connect(realign_run, 'out_file', joiner, 'bold_file')
        wf.connect(realign_run, 'par_file', joiner, 'realign_movpar')

    #realign across runs
    realign_all = realign_run.clone(name='realign_allruns')

    wf.connect(joiner, 'corrected_bolds', realign_all, 'in_file')
    wf.connect(realign_all, 'out_file', tsnr, 'in_file')
    wf.connect(tsnr, 'detrended_file', recalc_median, 'in_files')

    # segment and register
    if fs_dir:
        registration = create_fs_reg_workflow()
        registration.inputs.inputspec.subject_id = subject_id
        registration.inputs.inputspec.subjects_dir = fs_dir
        if target_file:
            registration.inputs.inputspec.target_image = target_file
        else:
            registration.inputs.inputspec.target_image = fsl.Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
    else:
        registration = create_reg_workflow()
        registration.inputs.inputspec.anatomical_image = anat
        registration.inputs.inputspec.target_image = fsl.Info.standard_image('MNI152_T1_2mm.nii.gz')
        registration.inputs.inputspec.config_file = 'T1_2_MNI152_2mm'
    wf.connect(recalc_median, 'median_file', registration, 'inputspec.mean_image')

    """ Quantify TSNR in each freesurfer ROI """
    get_roi_tsnr = MapNode(fs.SegStats(), iterfield=['in_file'], 
                           name='get_aparc_tsnr')
    get_roi_tsnr.inputs.default_color_table = True
    get_roi_tsnr.inputs.avgwf_txt_file = True
    wf.connect(tsnr, 'tsnr_file', get_roi_tsnr, 'in_file')
    wf.connect(registration, 'outputspec.aparc', get_roi_tsnr, 'segmentation_file')

    # Get a brain mask
    mask = Node(fsl.BET(), name='mask-bet')
    mask.inputs.mask = True
    wf.connect(recalc_median, 'median_file', mask, 'in_file')

    """ Detect outliers in a functional imaging series"""
    art = MapNode(ra.ArtifactDetect(),
                  iterfield=['realigned_files', 
                             'realignment_parameters'],
                  name="art")
    art.inputs.use_differences = [True, False]
    art.inputs.use_norm = True
    art.inputs.norm_threshold = 1
    art.inputs.zintensity_threshold = 3
    art.inputs.mask_type = 'spm_global'
    art.inputs.parameter_source = 'NiPy'
    wf.connect([(realign_all, art, [('out_file', 'realigned_files')]),
                (joiner, art, [('nipy_realign_pars', 'realignment_parameters')])
                ])

    def bandpass_filter(files, lowpass_freq, highpass_freq, fs):
        """Bandpass filter the input files
        Parameters
        ----------
        files: list of 4d nifti files
        lowpass_freq: cutoff frequency for the low pass filter (in Hz)
        highpass_freq: cutoff frequency for the high pass filter (in Hz)
        fs: sampling rate (in Hz)
        """
        out_files = []
        for filename in filename_to_list(files):
            path, name, ext = split_filename(filename)
            out_file = os.path.join(os.getcwd(), name + '_bp' + ext)
            img = nb.load(filename)
            timepoints = img.shape[-1]
            F = np.zeros((timepoints))
            lowidx = int(timepoints / 2) + 1
            if lowpass_freq > 0:
                lowidx = np.round(float(lowpass_freq) / fs * timepoints)
            highidx = 0
            if highpass_freq > 0:
                highidx = np.round(float(highpass_freq) / fs * timepoints)
            F[highidx:lowidx] = 1
            F = ((F + F[::-1]) > 0).astype(int)
            data = img.get_data()
            if np.all(F == 1):
                filtered_data = data
            else:
                filtered_data = np.real(np.fft.ifftn(np.fft.fftn(data) * F))
            img_out = nb.Nifti1Image(filtered_data, img.affine, img.header)
            img_out.to_filename(out_file)
            out_files.append(out_file)
        return list_to_filename(out_files)
    
    # Bandpass filters before registration
    bandpass = Node(Function(input_names=['files', 'lowpass_freq',
                                          'highpass_freq', 'fs'],
                             output_names=['out_files'],
                             function=bandpass_filter,
                             imports=imports),
                    name='bandpass_unsmooth')
    bandpass.inputs.fs = 1. / TR
    bandpass.inputs.highpass_freq = highpass_freq
    bandpass.inputs.lowpass_freq = lowpass_freq
    wf.connect(realign_all, 'out_file', bandpass, 'files')

    """Smooth the functional data using
    :class:`nipype.interfaces.fsl.IsotropicSmooth`.
    """
    smooth = MapNode(interface=fsl.IsotropicSmooth(), name="smooth", iterfield=["in_file"])
    smooth.inputs.fwhm = fwhm
    wf.connect(bandpass, 'out_files', smooth, 'in_file')
    # once highpassed and smoothed, pass into modelfit
    wf.connect(smooth, 'out_file', modelfit, 'inputspec.functional_data')

    def motion_regressors(motion_params, order=0, derivatives=1):
        """Compute motion regressors upto given order and derivative
        motion + d(motion)/dt + d2(motion)/dt2 (linear + quadratic)"""
        out_files = []
        for idx, filename in enumerate(filename_to_list(motion_params)):
            params = np.genfromtxt(filename)
            out_params = params
            for d in range(1, derivatives + 1):
                cparams = np.vstack((np.repeat(params[0, :][None, :], d, axis=0),
                                     params))
                out_params = np.hstack((out_params, np.diff(cparams, d, axis=0)))
            out_params2 = out_params
            for i in range(2, order + 1):
                out_params2 = np.hstack((out_params2, np.power(out_params, i)))
            filename = os.path.join(os.getcwd(), "motion_regressor%02d.txt" % idx)
            np.savetxt(filename, out_params2, fmt=str("%.10f"))
            out_files.append(filename)
        return out_files
    
    
    motreg = Node(Function(input_names=['motion_params', 'order',
                                        'derivatives'],
                           output_names=['out_files'],
                           function=motion_regressors,
                           imports=imports),
                  name='getmotionregress')
    wf.connect(joiner, 'nipy_realign_pars', motreg, 'motion_params')
    
    
    def build_filter1(motion_params, comp_norm, outliers, detrend_poly=None):
        """Builds a regressor set comprisong motion parameters, composite norm and
        outliers
        The outliers are added as a single time point column for each outlier
        Parameters
        ----------
        motion_params: a text file containing motion parameters and its derivatives
        comp_norm: a text file containing the composite norm
        outliers: a text file containing 0-based outlier indices
        detrend_poly: number of polynomials to add to detrend
        Returns
        -------
        components_file: a text file containing all the regressors
        """
        out_files = []
        for idx, filename in enumerate(filename_to_list(motion_params)):
            params = np.genfromtxt(filename)
            norm_val = np.genfromtxt(filename_to_list(comp_norm)[idx])
            out_params = np.hstack((params, norm_val[:, None]))
            try:
                outlier_val = np.genfromtxt(filename_to_list(outliers)[idx])
            except IOError:
                outlier_val = np.empty((0))
            for index in np.atleast_1d(outlier_val):
                outlier_vector = np.zeros((out_params.shape[0], 1))
                outlier_vector[index] = 1
                out_params = np.hstack((out_params, outlier_vector))
            if detrend_poly:
                timepoints = out_params.shape[0]
                X = np.empty((timepoints, 0))
                for i in range(detrend_poly):
                    X = np.hstack((X, legendre(
                        i + 1)(np.linspace(-1, 1, timepoints))[:, None]))
                out_params = np.hstack((out_params, X))
            filename = os.path.join(os.getcwd(), "filter_regressor%02d.txt" % idx)
            np.savetxt(filename, out_params, fmt=str("%.10f"))
            out_files.append(filename)
        return out_files

    # Create a filter to remove motion and art confounds
    createfilter1 = Node(Function(input_names=['motion_params', 'comp_norm',
                                               'outliers', 'detrend_poly'],
                                  output_names=['out_files'],
                                  function=build_filter1,
                                  imports=imports),
                            name='makemotionbasedfilter')
    createfilter1.inputs.detrend_poly = 2
    wf.connect(motreg, 'out_files', createfilter1, 'motion_params')
    wf.connect(art, 'norm_files', createfilter1, 'comp_norm')
    wf.connect(art, 'outlier_files', createfilter1, 'outliers')

    filter1 = MapNode(fsl.GLM(out_f_name='F_mcart.nii.gz',
                              out_pf_name='pF_mcart.nii.gz',
                              demean=True),
                      iterfield=['in_file', 'design', 'out_res_name'],
                      name='filtermotion')
    wf.connect(realign_all, 'out_file', filter1, 'in_file')
    wf.connect(realign_all, ('out_file', rename, '_filtermotart'), 
               filter1, 'out_res_name')
    wf.connect(createfilter1, 'out_files', filter1, 'design')

    def selectindex(files, idx):
        """ Utility function for registration seg files """
        import numpy as np
        from nipype.utils.filemanip import filename_to_list, list_to_filename
        return list_to_filename(np.array(filename_to_list(files))[idx].tolist())

        
    def extract_noise_components(realigned_file, mask_file, num_components=5,
                                 extra_regressors=None):
        """Derive components most reflective of physiological noise
        Parameters
        ----------
        realigned_file: a 4D Nifti file containing realigned volumes
        mask_file: a 3D Nifti file containing white matter + ventricular masks
        num_components: number of components to use for noise decomposition
        extra_regressors: additional regressors to add
        Returns
        -------
        components_file: a text file containing the noise components
        """
        imgseries = nb.load(realigned_file)
        components = None
        for filename in filename_to_list(mask_file):
            mask = nb.load(filename).get_data()
            if len(np.nonzero(mask > 0)[0]) == 0:
                continue
            voxel_timecourses = imgseries.get_data()[mask > 0]
            voxel_timecourses[np.isnan(np.sum(voxel_timecourses, axis=1)), :] = 0
            # remove mean and normalize by variance
            # voxel_timecourses.shape == [nvoxels, time]
            X = voxel_timecourses.T
            stdX = np.std(X, axis=0)
            stdX[stdX == 0] = 1.
            stdX[np.isnan(stdX)] = 1.
            stdX[np.isinf(stdX)] = 1.
            X = (X - np.mean(X, axis=0)) / stdX
            u, _, _ = sp.linalg.svd(X, full_matrices=False)
            if components is None:
                components = u[:, :num_components]
            else:
                components = np.hstack((components, u[:, :num_components]))
        if extra_regressors:
            regressors = np.genfromtxt(extra_regressors)
            components = np.hstack((components, regressors))
        components_file = os.path.join(os.getcwd(), 'noise_components.txt')
        np.savetxt(components_file, components, fmt=str("%.10f"))
        return components_file

    createfilter2 = MapNode(Function(input_names=['realigned_file',
                                                  'mask_file',
                                                  'num_components',
                                                  'extra_regressors'],
                                     output_names=['out_files'],
                                     function=extract_noise_components,
                                     imports=imports),
                            iterfield=['realigned_file', 'extra_regressors'],
                            name='makecompcorrfilter')
    wf.connect(createfilter1, 'out_files', createfilter2, 'extra_regressors')
    wf.connect(filter1, 'out_res', createfilter2, 'realigned_file')
    wf.connect(registration, ('outputspec.segmentation_files', selectindex, [0, 2]),
               createfilter2, 'mask_file')

    filter2 = MapNode(fsl.GLM(out_f_name='F.nii.gz',
                              out_pf_name='pF.nii.gz',
                              demean=True),
                      iterfield=['in_file', 'design', 'out_res_name'],
                      name='filter_noise_nosmooth')
    wf.connect(filter1, 'out_res', filter2, 'in_file')
    wf.connect(filter1, ('out_res', rename, '_cleaned'),
               filter2, 'out_res_name')
    wf.connect(createfilter2, 'out_files', filter2, 'design')
    wf.connect(mask, 'mask_file', filter2, 'mask')

    # Do it again to check
    bandpass2 = bandpass.clone(name='bp2_unsmooth')
    wf.connect(filter2, 'out_res', bandpass2, 'files')
    # Do again for check
    smooth2 = smooth.clone(name="smooth2")
    wf.connect(bandpass2, 'out_files', smooth2, 'in_file')

    collector = Node(Merge(2), name='collect_streams')
    wf.connect(smooth2, 'out_file', collector, 'in1')
    wf.connect(bandpass2, 'out_files', collector, 'in2')

    """
    Transform the remaining images. First to anatomical and then to target
    """

    warpall = MapNode(ants.ApplyTransforms(), iterfield=['input_image'],
                      name='warpall')
    warpall.inputs.input_image_type = 3
    warpall.inputs.interpolation = 'Linear'
    warpall.inputs.invert_transform_flags = [False, False]
    warpall.inputs.terminal_output = 'file'
    warpall.inputs.reference_image = target_file
    warpall.inputs.args = '--float'
    warpall.inputs.num_threads = 2
    warpall.plugin_args = {'sbatch_args': '-c%d' % 2}

    # transform to target
    wf.connect(collector, 'out', warpall, 'input_image')
    wf.connect(registration, 'outputspec.transforms', warpall, 'transforms')

    mask_target = Node(fsl.ImageMaths(op_string='-bin'), name='target_mask')

    wf.connect(registration, 'outputspec.anat2target', mask_target, 'in_file')

    maskts = MapNode(fsl.ApplyMask(), iterfield=['in_file'], name='ts_masker')
    wf.connect(warpall, 'output_image', maskts, 'in_file')
    wf.connect(mask_target, 'out_file', maskts, 'mask_file')

    # BOLD modeling
    def get_contrasts(contrast_file, task_id, conds):
        """ Setup a basic set of contrasts, a t-test per condition """
        import numpy as np
        import os
        contrast_def = []
        if os.path.exists(contrast_file):
            with open(contrast_file, 'rt') as fp:
                contrast_def.extend([np.array(row.split()) for row in fp.readlines() if row.strip()])
        contrasts = []
        for row in contrast_def:
            if row[0] != 'task-%s' % task_id:
                continue
            con = [row[1], 'T', ['cond%03d' % (i + 1)  for i in range(len(conds))],
                   row[2:].astype(float).tolist()]
            contrasts.append(con)
        # add auto contrasts for each column
        for i, cond in enumerate(conds):
            con = [cond, 'T', ['cond%03d' % (i + 1)], [1]]
            contrasts.append(con)
        return contrasts

    contrastgen = Node(Function(input_names=['contrast_file',
                                             'task_id', 'conds'],
                                output_names=['contrasts'],
                                function=get_contrasts),
                       name='contrastgen')
    contrastgen.inputs.contrast_file = contrast
    contrastgen.inputs.task_id = task_id
    contrastgen.inputs.conds = conds
    wf.connect(contrastgen, 'contrasts', modelfit, 'inputspec.contrasts')

    def check_behav_list(behav, run_id, conds):
        """ Check and reshape cond00x.txt files """
        import six
        import numpy as np
        num_conds = len(conds)
        if isinstance(behav, six.string_types):
            behav = [behav]
        behav_array = np.array(behav).flatten()
        num_elements = behav_array.shape[0]
        return behav_array.reshape(num_elements/num_conds, num_conds).tolist()

    reshape_behav = Node(Function(input_names=['behav', 'run_id', 'conds'],
                                  output_names=['behav'],
                                  function=check_behav_list),
                         name='reshape_behav')
    reshape_behav.inputs.behav = behav
    reshape_behav.inputs.run_id = run_id
    reshape_behav.inputs.conds = conds

    modelspec = Node(model.SpecifyModel(),
                     name="modelspec")
    modelspec.inputs.input_units = 'secs'
    modelspec.inputs.time_repetition = TR
    # in seconds
    modelspec.inputs.high_pass_filter_cutoff = (1./highpass_freq)

    # bold model connections
    wf.connect(reshape_behav, 'behav', modelspec, 'event_files')
    wf.connect(realign_all, 'out_file', modelspec, 'functional_runs')
    wf.connect(joiner, 'nipy_realign_pars', modelspec, 'realignment_parameters')
    wf.connect(art, 'outlier_files', modelspec, 'outlier_files')
    wf.connect(modelspec, 'session_info', modelfit, 'inputspec.session_info')
    # OLD METHOD
    #wf.connect(realign_all, 'out_file', modelfit, 'inputspec.functional_data')

    def sort_copes(copes, varcopes, contrasts):
        """Reorder the copes so that now it combines across runs"""
        import numpy as np
        if not isinstance(copes, list):
            copes = [copes]
            varcopes = [varcopes]
        num_copes = len(contrasts)
        n_runs = len(copes)
        all_copes = np.array(copes).flatten()
        all_varcopes = np.array(varcopes).flatten()
        outcopes = all_copes.reshape(len(all_copes)/num_copes, num_copes).T.tolist()
        outvarcopes = all_varcopes.reshape(len(all_varcopes)/num_copes, num_copes).T.tolist()
        return outcopes, outvarcopes, n_runs

    cope_sorter = Node(Function(input_names=['copes', 'varcopes',
                                             'contrasts'],
                                output_names=['copes', 'varcopes',
                                              'n_runs'],
                                function=sort_copes),
                       name='cope_sorter')

    pickfirst = lambda x: x[0]

    wf.connect(contrastgen, 'contrasts', cope_sorter, 'contrasts')
    wf.connect([(mask, fixed_fx, [('mask_file', 'flameo.mask_file')]),
                (modelfit, cope_sorter, [('outputspec.copes', 'copes')]),
                (modelfit, cope_sorter, [('outputspec.varcopes', 'varcopes')]),
                (cope_sorter, fixed_fx, [('copes', 'inputspec.copes'),
                                         ('varcopes', 'inputspec.varcopes'),
                                         ('n_runs', 'l2model.num_copes')]),
                (modelfit, fixed_fx, [('outputspec.dof_file',
                                        'inputspec.dof_files')])])

    def merge_files(copes, varcopes, zstats):
        out_files = []
        splits = []
        out_files.extend(copes)
        splits.append(len(copes))
        out_files.extend(varcopes)
        splits.append(len(varcopes))
        out_files.extend(zstats)
        splits.append(len(zstats))
        return out_files, splits

    mergefunc = Node(Function(input_names=['copes', 'varcopes',
                                                  'zstats'],
                                   output_names=['out_files', 'splits'],
                                   function=merge_files),
                      name='merge_files')
    wf.connect([(fixed_fx.get_node('outputspec'), mergefunc,
                                 [('copes', 'copes'),
                                  ('varcopes', 'varcopes'),
                                  ('zstats', 'zstats'),
                                  ])])
    wf.connect(mergefunc, 'out_files', registration, 'inputspec.source_files')

    def split_files(in_files, splits):
        copes = in_files[:splits[0]]
        varcopes = in_files[splits[0]:(splits[0] + splits[1])]
        zstats = in_files[(splits[0] + splits[1]):]
        return copes, varcopes, zstats

    splitfunc = Node(Function(input_names=['in_files', 'splits'],
                                     output_names=['copes', 'varcopes',
                                                   'zstats'],
                                     function=split_files),
                      name='split_files')
    wf.connect(mergefunc, 'splits', splitfunc, 'splits')
    wf.connect(registration, 'outputspec.transformed_files',
               splitfunc, 'in_files')

    if fs_dir:
        get_roi_mean = MapNode(fs.SegStats(default_color_table=True),
                                  iterfield=['in_file'], name='get_aparc_means')
        get_roi_mean.inputs.avgwf_txt_file = True
        wf.connect(fixed_fx.get_node('outputspec'), 'copes', get_roi_mean, 'in_file')
        wf.connect(registration, 'outputspec.aparc', get_roi_mean, 'segmentation_file')

        # Sample the average time series in aparc ROIs
        # from rsfmri_vol_surface_preprocessing_nipy.py
        sampleaparc = MapNode(fs.SegStats(default_color_table=True),
	                          iterfield=['in_file'],
	                          name='aparc_ts')
        sampleaparc.inputs.segment_id = ([8] + range(10, 14) + [17, 18, 26, 47] +
	                                     range(49, 55) + [58] + range(1001, 1036) +
	                                     range(2001, 2036))
        sampleaparc.inputs.avgwf_txt_file = True
	
        wf.connect(registration, 'outputspec.aparc', sampleaparc, 'segmentation_file')
        wf.connect(realign_all, 'out_file', sampleaparc, 'in_file')

    def get_subs(subject_id, conds, run_id, model_id, task_id):
        """ Substitutions for files saved in datasink """
        subs = [('_subject_id_%s_' % subject_id, '')]
        subs.append(('_model_id_%d' % model_id, 'model%03d' %model_id))
        subs.append(('task_id_%s/' % task_id, '/task-%s_' % task_id))
        subs.append(('bold_dtype_mcf_mask_smooth_mask_gms_tempfilt_mean_warp',
        'mean'))
        subs.append(('bold_dtype_mcf_mask_smooth_mask_gms_tempfilt_mean_flirt',
        'affine'))
        # substitutions per condition
        for i in range(len(conds)):
            subs.append(('_flameo%d/cope1.' % i, 'cope%02d.' % (i + 1)))
            subs.append(('_flameo%d/varcope1.' % i, 'varcope%02d.' % (i + 1)))
            subs.append(('_flameo%d/zstat1.' % i, 'zstat%02d.' % (i + 1)))
            subs.append(('_flameo%d/tstat1.' % i, 'tstat%02d.' % (i + 1)))
            subs.append(('_flameo%d/res4d.' % i, 'res4d%02d.' % (i + 1)))
            subs.append(('_warpall%d/cope1_warp.' % i,
                         'cope%02d.' % (i + 1)))
            subs.append(('_warpall%d/varcope1_warp.' % (len(conds) + i),
                         'varcope%02d.' % (i + 1)))
            subs.append(('_warpall%d/zstat1_warp.' % (2 * len(conds) + i),
                         'zstat%02d.' % (i + 1)))
            subs.append(('_warpall%d/cope1_trans.' % i,
                         'cope%02d.' % (i + 1)))
            subs.append(('_warpall%d/varcope1_trans.' % (len(conds) + i),
                         'varcope%02d.' % (i + 1)))
            subs.append(('_warpall%d/zstat1_trans.' % (2 * len(conds) + i),
                         'zstat%02d.' % (i + 1)))
            subs.append(('__get_aparc_means%d/' % i, '/cope%02d_' % (i + 1)))
        # substitutions per run
        for i, run_num in enumerate(run_id):
            subs.append(('__get_aparc_tsnr%d/' % i, '/run%02d_' % run_num))
            subs.append(('__art%d/' % i, '/run%02d_' % run_num))
            subs.append(('__dilatemask%d/' % i, '/run%02d_' % run_num))
            subs.append(('__realign%d/' % i, '/run%02d_' % run_num))
            subs.append(('__modelgen%d/' % i, '/run%02d_' % run_num))

        subs.append(('/%s/%s/' % (task_id, subject_id), '/task-%s/' % task_id))
        subs.append(('/model%03d/task-%s_' % (model_id, task_id), '/'))
        subs.append(('_bold_dtype_mcf_bet_thresh_dil', '_mask'))
        subs.append(('mask/model%03d/task-%s/' % (model_id, task_id), 'mask/'))
        subs.append(('tsnr/model%03d/task-%s/' % (model_id, task_id), 'tsnr/'))
        subs.append(('_output_warped_image', '_anat2target'))
        subs.append(('median_flirt_brain_mask', 'median_brain_mask'))
        subs.append(('median_bbreg_brain_mask', 'median_brain_mask'))
        return subs

    subsgen = Node(Function(input_names=['subject_id', 'conds', 'run_id',
                                                'model_id', 'task_id'],
                                   output_names=['substitutions'],
                                   function=get_subs),
                      name='subsgen')
    subsgen.inputs.subject_id = subject_id
    subsgen.inputs.model_id = model_id
    subsgen.inputs.task_id = task_id
    subsgen.inputs.run_id = run_id

    # Sink data of interest
    datasink = Node(DataSink(), name="datasink")
    datasink.inputs.container = subject_id

    wf.connect(contrastgen, 'contrasts', subsgen, 'conds')
    wf.connect(subsgen, 'substitutions', datasink, 'substitutions')
    wf.connect([(fixed_fx.get_node('outputspec'), datasink,
                                 [('res4d', 'res4d'),
                                  ('copes', 'copes'),
                                  ('varcopes', 'varcopes'),
                                  ('zstats', 'zstats'),
                                  ('tstats', 'tstats')])
                                 ])
    wf.connect([(modelfit.get_node('modelgen'), datasink,
                                 [('design_cov', 'qa.model'),
                                  ('design_image', 'qa.model.@matrix_image'),
                                  ('design_file', 'qa.model.@matrix'),
                                 ])])
    wf.connect([(joiner, datasink, [('nipy_realign_pars',
                                      'qa.motion')]),
                (mask, datasink, [('mask_file', 'qa.mask')])])
    wf.connect(art, 'norm_files', datasink, 'qa.art.@norm')
    wf.connect(art, 'intensity_files', datasink, 'qa.art.@intensity')
    wf.connect(art, 'outlier_files', datasink, 'qa.art.@outlier_files')
    wf.connect(registration, 'outputspec.anat2target', datasink, 'qa.anat2target')
    wf.connect(tsnr, 'tsnr_file', datasink, 'qa.tsnr.@map')
    if fs_dir:
        wf.connect(registration, 'outputspec.min_cost_file', datasink, 'qa.mincost')
        wf.connect([(get_roi_tsnr, datasink, [('avgwf_txt_file', 'qa.tsnr'),
                                              ('summary_file', 'qa.tsnr.@summary')])])
        wf.connect([(get_roi_mean, datasink, [('avgwf_txt_file', 'copes.roi'),
                                              ('summary_file', 'copes.roi.@summary')])])
        wf.connect(sampleaparc, 'summary_file', datasink, 'timeseries.aparc.@summary')
        wf.connect(sampleaparc, 'avgwf_txt_file', datasink, 'timeseries.aparc')
    wf.connect([(splitfunc, datasink,
                 [('copes', 'copes.mni'),
                  ('varcopes', 'varcopes.mni'),
                  ('zstats', 'zstats.mni'),
                  ])])
    wf.connect(recalc_median, 'median_file', datasink, 'mean')
    wf.connect(registration, 'outputspec.transformed_mean', datasink, 'mean.mni')
    wf.connect(registration, 'outputspec.func2anat_transform', datasink, 'xfm.mean2anat')
    wf.connect(registration, 'outputspec.anat2target_transform', datasink, 'xfm.anat2target')

    """
    Set processing parameters
    """

    #modelspec.inputs.high_pass_filter_cutoff = hpcutoff
    modelfit.inputs.inputspec.bases = {'dgamma': {'derivs': use_derivatives}}
    modelfit.inputs.inputspec.model_serial_correlations = True
    modelfit.inputs.inputspec.film_threshold = 1000

    datasink.inputs.base_directory = outdir
    return wf

"""
The following functions run the whole workflow.
"""

if __name__ == '__main__':
    import argparse
    defstr = ' (default %(default)s)'
    parser = argparse.ArgumentParser(prog='fmri_openfmri.py',
                                     description=__doc__)
    parser.add_argument('-d', '--datasetdir', required=True)
    parser.add_argument('-s', '--subject', default=[],
                        nargs='+', type=str,
                        help="Subject name (e.g. 'sub001')")
    parser.add_argument('-m', '--model', default=1, type=int,
                        help="Model index" + defstr)
    parser.add_argument('-x', '--subjectprefix', default='sub*',
                        help="Subject prefix" + defstr)
    parser.add_argument('-t', '--task', required=True, #nargs='+',
                        type=str, help="Task name" + defstr)
    parser.add_argument('--hpfilter', default=0.01, type=float, 
                        help="High pass frequency (Hz)" + defstr)
    parser.add_argument('--lpfilter', default=0.1, type=float,
                        help="Low pass frequency (Hz)" + defstr)
    parser.add_argument('--fwhm', default=6.,
                        type=float, help="Spatial FWHM" + defstr)
    parser.add_argument('--derivatives', action="store_true",
                        help="Use derivatives" + defstr)
    parser.add_argument("-o", "--output_dir", dest="outdir",
                        help="Output directory base")
    parser.add_argument("-w", "--work_dir", dest="work_dir",
                        help="Output directory base")
    parser.add_argument("-p", "--plugin", dest="plugin",
                        default='Linear',
                        help="Plugin to use")
    parser.add_argument("--plugin_args", dest="plugin_args",
                        help="Plugin arguments")
    parser.add_argument("--sd", dest="fs_dir", default=None,
                        help="FreeSurfer subjects directory (if available)")
    parser.add_argument("--target", dest="target_file",
                        help=("Target in MNI space. Best to use the MindBoggle "
                              "template - only used with FreeSurfer"
                              "OASIS-30_Atropos_template_in_MNI152_2mm.nii.gz"))
    parser.add_argument("--session_id", dest="session_id", default=None,
                        help="Session id, ex. 'ses-1'")
    parser.add_argument("--crashdump_dir", dest="crashdump_dir",
                        help="Crashdump dir", default=None)
    parser.add_argument('--topup', action="store_true",
                        help="Apply topup correction" + defstr)
    parser.add_argument('--debug', action="store_true",
                        help="Activate nipype debug mode" + defstr)
    args = parser.parse_args()
    
    data_dir = os.path.abspath(args.datasetdir)
    if args.work_dir:
        workdir = os.path.abspath(args.work_dir)
    else:
        workdir = os.getcwd()
    if args.outdir:
        outdir = os.path.abspath(args.outdir)
    else:
        outdir = os.path.join(workdir, 'output')
    if args.crashdump_dir:
        crashdump = os.path.abspath(args.crashdump_dir)
        wf.config['execution']['crashdump_dir'] = crashdump
    else:
        crashdump = os.getcwd()

    derivatives = args.derivatives
    if derivatives is None:
       derivatives = False

    fs_dir = args.fs_dir
    if fs_dir is not None:
        fs_dir = os.path.abspath(fs_dir)

    if args.debug:
        from nipype import logging
        config.enable_debug_mode()
        logging.update_logging(config)
        
    wf = create_workflow(data_dir, args, 
                         fs_dir, derivatives,
                         workdir, outdir)
    wf.base_dir = workdir
    
    # Optional changes
    #wf.config['execution']['remove_unnecessary_outputs'] = False
    #wf.config['execution']['poll_sleep_duration'] = 2
    
    if args.plugin_args:
        wf.run(args.plugin, plugin_args=eval(args.plugin_args))
    else:
        wf.run(args.plugin)
