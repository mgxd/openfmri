"""
Individual and (group level?) analysis using `fmriprep` outputs and FSL.

Adapted script from original notebook:
https://bit.ly/2Scu6AT

Requirement: BIDS dataset (including events.tsv), 
fmriprep outputs, and modeling files [more on this when concrete]

TODO: use pybids derivatives fetcher
"""
import os
import os.path as op
import argparse

try:
    from bids.layout import BIDSLayout
except ImportError:
    from bids.grabbids import BIDSLayout

import numpy as np
import nibabel as nb

import nipype.algorithms.modelgen as model
from nipype.interfaces.base import Bunch
from nipype import (Workflow, Node, MapNode, IdentityInterface, Function, 
                    DataSink, JoinNode, Merge)
from nipype.interfaces import spm, fsl
from nipype.workflows.fmri.fsl import create_fixed_effects_flow

__version__ = '0.0.2'

IdentityNode = lambda fields, name: Node(IdentityInterface(fields=fields), name=name)

# TODO: HTML reporting similar to fmriprep
def create_prep_l1_wf(bids_dir, model, subj, task, fmriprep_dir, TR, regressors,
                      session, dropvols, fwhm, highpass, lowpass,
                      name="prep_wf"):
    """
    Preparation workflow for subject level analysis.
    Generates modeling and optionally applies the 
    following steps to the preprocessed functional:
        * removal of initial volumes
        * smoothing
        * bandpass filtering
    
    Parameters
    ==========
    bids_dir
    model
    subj
    task
    fmriprep_dir
    outdir
    session
    TR
    dropvols
    fwhm
    highpass
    lowpass
    
    Returns
    =======
    wf
    
    wf.inputnode.inputs
    ===================
    run_id
    event_file
    regressors
    
    wf.outputnode.outputs
    =====================
    func
    info - Bunch
    contrasts
    
    """
    # initialize workflow
    wf = Workflow(name=name.format(subj, task))

    inputnode = IdentityNode(['run_id', 'event_file'],
                             'inputnode')
    def get_data(subj, task, fmriprep_dir, run_id, session=None):
        """Quick filegrabber ala SelectFiles/DataGrabber"""
        import os.path as op
        from glob import glob

        pattern = 'sub-{}*task-{}*run*{}'.format(subj, task, run_id)
        fmriprep_func = op.join(fmriprep_dir, "sub-{}".format(subj), "func")
        if session:
            fmriprep_func = op.join(fmriprep_dir, "sub-{}".format(subj), 
                                    "ses-{}".format(session), "func")
        # grab these files
        confound_file = glob(op.join(fmriprep_func, "{}*confounds.tsv".format(pattern)))[0]
        mni_file = glob(op.join(
            fmriprep_func, 
            "{}*space-MNI152NLin2009cAsym_preproc.nii.gz".format(pattern)
        ))[0]
        mni_mask = mni_file.replace("_preproc.", "_brainmask.")
        return confound_file, mni_file, mni_mask


    datagrabber = Node(Function(output_names=["confound_file", 
                                              "mni_file", 
                                              "mni_mask"],
                               function=get_data), 
                      name='datagrabber')
    datagrabber.inputs.subj = subj
    datagrabber.inputs.task = task
    datagrabber.inputs.fmriprep_dir = fmriprep_dir
    datagrabber.inputs.session = session
    wf.connect(inputnode, "run_id", datagrabber, "run_id")

    def gen_model_info(event_file, confound_file, regressor_names, dropvols):
        """Defines `SpecifyModel` information from BIDS events."""
        import pandas as pd
        from nipype.interfaces.base import Bunch

        events = pd.read_csv(event_file, sep="\t")
        trial_types = events.trial_type.unique()
        onset = []
        duration = []
        regressors = []
        for trial in trial_types:
            onset.append(events[events.trial_type == trial].onset.tolist())
            duration.append(events[events.trial_type == trial].duration.tolist())

        confounds = pd.read_csv(confound_file, sep="\t", na_values="n/a")
        for regressor in regressor_names:
            if regressor == 'FramewiseDisplacement':
                regressors.append(confounds[regressor].fillna(0).tolist()[dropvols:])
            else:
                regressors.append(confounds[regressor].tolist()[dropvols:])

        info = [Bunch(
            conditions=trial_types,
            onsets=onset,
            durations=duration,
            regressors=regressors,
            regressor_names=regressor_names,
        )]
        return info


    modelinfo = Node(Function(output_names=['info'],
                              function=gen_model_info), 
                     name="modelinfo")
    modelinfo.inputs.dropvols = dropvols
    modelinfo.inputs.regressor_names = regressors

    # these will likelybe in bids model json in the future
    # for now, replace with argparse argument
    wf.connect(inputnode, "event_file", modelinfo, "event_file")
    wf.connect(datagrabber, "confound_file", modelinfo, "confound_file")

    roi = Node(fsl.ExtractROI(t_min=dropvols, t_size=-1), name="extractroi")
    wf.connect(datagrabber, "mni_file", roi, "in_file")

    # smooth
    if fwhm:
        smooth = Node(fsl.IsotropicSmooth(), name='smooth')
        smooth.inputs.fwhm = fwhm
        wf.connect(roi, 'roi_file', smooth, 'in_file')

    def bandpass_filter(in_file, lowpass, highpass, fs):
        """Bandpass filter the input file

        Parameters
        ----------
        file: 4D nifti file   
        lowpass: cutoff (sec) for low pass filter (0 to not filter)
        highpass: cutoff (sec) for high pass filter (0 to not filter)
        fs: sampling rate (1/TR)
        """
        import nibabel as nb
        import numpy as np
        import os
        from nipype.utils.filemanip import split_filename

        path, name, ext = split_filename(in_file)
        out_file = os.path.join(os.getcwd(), name + '_bp' + ext)

        img = nb.load(in_file)
        if len(img.shape) != 4:
            raise RuntimeError("Input is not a 4D NIfTI file")
        vols = img.shape[-1]
        lidx = int(np.round((1. / lowpass) / fs * vols) if lowpass > 0
                   else vols // 2 + 1)
        hidx = int(np.round((1. / highpass) / fs * vols) if highpass > 0
                   else 0)
        F = np.zeros((vols))
        F[hidx:lidx] = 1
        F = ((F + F[::-1]) > 0).astype(int)
        if np.all(F == 1):
            filtered_data = img.get_data()
        else:
            filtered_data = np.real(np.fft.ifftn(np.multiply(np.fft.fftn(img.get_data()), F)))

        img_out = img.__class__(filtered_data, 
                                img.affine,
                                img.header)
        img_out.to_filename(out_file)
        return out_file


    if (highpass or lowpass) and (model != 'spm'):
        bandpass = Node(Function(function=bandpass_filter), name='bandpass')
        bandpass.inputs.highpass = highpass
        bandpass.inputs.lowpass = lowpass
        bandpass.inputs.fs = 1./TR

        # TODO: generalize this
        if fwhm:
            wf.connect(smooth, 'out_file', bandpass, 'in_file')
        else:
            wf.connect(roi, 'roi_file', bandpass, 'in_file')

    def read_contrasts(bids_dir, task):
        """potential BUG? This will not update if contrasts file is changed.
        should be fixed with config option to skip hash check"""
        import os.path as op

        contrasts = []
        contrasts_file = op.join(bids_dir, "code", "contrasts.tsv")

        if not op.exists(contrasts_file):
            raise FileNotFoundError("Contrasts file not found.")

        with open(contrasts_file, "r") as fp:
            info = [line.strip().split("\t") for line in fp.readlines()]

        for row in info:
            if row[0] != task:
                continue

            contrasts.append([
                row[1], 
                "T", 
                [cond for cond in row[2].split(" ")],
                [float(w) for w in row[3].split(" ")]
            ])
        if not contrasts:
            raise AttributeError("No contrasts found for task {}".format(task))
        return contrasts


    contrastgen = Node(Function(output_names=["contrasts"],
                                function=read_contrasts),
                       name="contrastgen")
    contrastgen.inputs.bids_dir = bids_dir
    contrastgen.inputs.task = task
    
    #cfg = dict(execution={'local_hash_check': False})
    #contrastgen.config.update_config(cfg)

    outputnode = IdentityNode(['func', 'info', 'contrasts', 'mask'],
                              'outputnode')

    ## Split off into individual workflows
    if highpass and (model != 'spm'):
        wf.connect(bandpass, 'out', outputnode, 'func')
    elif fwhm:
        wf.connect(smooth, 'out_file', outputnode, 'func')
    else:
        wf.connect(roi, "roi_file", outputnode, "func")

    wf.connect(contrastgen, 'contrasts', outputnode, 'contrasts')
    wf.connect(modelinfo, 'info', outputnode, 'info')
    wf.connect(datagrabber, 'mni_mask', outputnode, 'mask')
    return wf


def create_spm_l1_wf(TR, highpass, name='spm_wf'):
    """"""
    wf = Workflow(name=name)
    inputnode = IdentityNode(['funcs', 'infos', 'contrasts'],
                             'inputnode')

    # TODO: check if func is compressed; if so, uncompress
    def uncompresser(func):
        if func.endswith('.gz'): # unzip
            from nipype.algorithms.misc import Gunzip
            res = Gunzip(in_file=func).run()
            return res.outputs.out_file
        return func
    
    uncompress = MapNode(Function(function=uncompresser, output_names=['funcs']),
                         iterfield=['func'],
                         name='uncompress')
    wf.connect(inputnode, 'funcs', uncompress, 'func')

    modelspec = Node(model.SpecifySPMModel(), name="modelspec")
    modelspec.inputs.concatenate_runs = False
    modelspec.inputs.output_units = "secs"
    modelspec.inputs.input_units = "secs"
    modelspec.inputs.time_repetition = TR
    modelspec.inputs.high_pass_filter_cutoff = highpass
    wf.connect(inputnode, "infos", modelspec, "subject_info")
    wf.connect(uncompress, "funcs", modelspec, "functional_runs")

    level1design = Node(spm.Level1Design(), name="level1design")
    level1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
    level1design.inputs.interscan_interval = TR
    level1design.inputs.timing_units = 'secs'
    wf.connect(modelspec, "session_info", level1design, "session_info")

    modelest = Node(spm.EstimateModel(), name='estimatemodel')
    modelest.inputs.estimation_method = {"Classical": 1}
    wf.connect(level1design, "spm_mat_file", modelest, "spm_mat_file")

    contrastest = Node(spm.EstimateContrast(), name='estimatecontrast')
    wf.connect(inputnode, "contrasts", contrastest, "contrasts")
    wf.connect(modelest, "spm_mat_file", contrastest, "spm_mat_file")
    wf.connect(modelest, "beta_images", contrastest, "beta_images")
    wf.connect(modelest, "residual_image", contrastest, "residual_image")

    outputnode = IdentityNode(['mat_file', 'con_images', 'spmT_images'], 
                              'outputnode')
    wf.connect(contrastest, 'spmT_images', outputnode, 'spmT_images')
    wf.connect(contrastest, 'con_images', outputnode, 'con_images')
    wf.connect(contrastest, 'spm_mat_file', outputnode, 'mat_file')

    return wf


def create_fsl_l1_wf(runs, highpass, TR, TA=None, name='fsl_wf'):
    """"""
    wf = Workflow(name=name)
    
    inputnode = Node(IdentityInterface(fields=['func', 'info', 'contrasts', 'mask']), 
                     name='inputnode')
    
    if TA:
        modelspec = Node(model.SpecifySparseModel(), name="modelspec")
        modelspec.inputs.time_acquisition = TA
    else:
        modelspec = Node(model.SpecifyModel(), name="modelspec")
    modelspec.inputs.input_units = "secs"
    modelspec.inputs.time_repetition = TR
    modelspec.inputs.high_pass_filter_cutoff = highpass
    wf.connect(inputnode, "info", modelspec, "subject_info")
    wf.connect(inputnode, "func", modelspec, "functional_runs")

    level1design = Node(fsl.Level1Design(), name="level1design")
    level1design.inputs.bases = {"dgamma": {"derivs": True}}
    level1design.inputs.model_serial_correlations = True
    level1design.inputs.interscan_interval = TR
    wf.connect(inputnode, "contrasts", level1design, "contrasts")
    wf.connect(modelspec, "session_info", level1design, "session_info")

    modelgen = Node(fsl.FEATModel(), name="modelgen")
    wf.connect(level1design, "fsf_files", modelgen, "fsf_file")
    wf.connect(level1design, "ev_files", modelgen, "ev_files")

    # mask GLM - maybe do before model split
    masker = Node(fsl.ApplyMask(), name="masker")
    wf.connect(inputnode, "mask", masker, "mask_file")
    wf.connect(inputnode, "func", masker, "in_file")

    glm = Node(fsl.FILMGLS(), name="filmgls")
    glm.inputs.autocorr_noestimate = True
    wf.connect(masker, "out_file", glm, "in_file")
    wf.connect(modelgen, "design_file", glm, "design_file")
    wf.connect(modelgen, "con_file", glm, "tcon_file")
    wf.connect(modelgen, "fcon_file", glm, "fcon_file")
    
    outputnode = IdentityNode(['mni_mask', 'dof_file', 'copes', 'varcopes'],
                              'outputnode')

    wf.connect(inputnode, 'mask', outputnode, 'mni_mask')
    wf.connect(glm, 'dof_file', outputnode, 'dof_file')
    wf.connect(glm, 'copes', outputnode, 'copes')
    wf.connect(glm, 'varcopes', outputnode, 'varcopes')
    
    return wf


def create_l1_wf(bids_dir, model, subj, task, fmriprep_dir, runs, regressors,
                 outdir, events, session, TR, workdir, dropvols, fwhm, 
                 highpass, lowpass, TA=None, name="l1_{}_task-{}_wf"):
    """
    Subject level analysis using `fmriprep` output
    """
    wf = Workflow(name=name.format(subj, task),
                  base_dir=workdir)

    # iter across runs
    inputnode = IdentityNode(['run_id', 'event_file'],
                             'inputnode')
    inputnode.iterables = [('run_id', runs),
                            ('event_file', events)]
    inputnode.synchronize = True


    prep_wf = create_prep_l1_wf(bids_dir, model, subj, task, fmriprep_dir, TR, regressors,
                                session, dropvols, fwhm, highpass, lowpass)
    
    wf.connect(inputnode, 'run_id', prep_wf, 'inputnode.run_id')
    wf.connect(inputnode, 'event_file', prep_wf, 'inputnode.event_file')

    pickfirst = lambda x: x[0] if not isinstance(x, str) else x

    if model == 'spm':
        
        # join, then model
        joinfields = ['infos', 'funcs']
        joiner = JoinNode(IdentityInterface(fields=joinfields), 
                          joinsource='inputnode',
                          joinfield=joinfields,
                          name='joiner-spm')
        wf.connect(prep_wf, 'outputnode.info', joiner, 'infos')
        wf.connect(prep_wf, 'outputnode.func', joiner, 'funcs')

        # merge info Bunches
        infomerge = Node(Merge(1), name='infomerge')
        infomerge.inputs.ravel_inputs = True
        wf.connect(joiner, 'infos', infomerge, 'in1')

        model_wf = create_spm_l1_wf(TR, highpass)
        wf.connect(infomerge, 'out', model_wf, 'inputnode.infos')
        wf.connect(joiner, 'funcs', model_wf, 'inputnode.funcs')
        wf.connect(prep_wf, 'outputnode.contrasts', model_wf, 'inputnode.contrasts')
        
    # continue iterating across runs
    elif model == 'fsl':
        model_wf = create_fsl_l1_wf(runs, highpass, TR, TA)
        wf.connect(prep_wf, 'outputnode.info', model_wf, 'inputnode.info')
        wf.connect(prep_wf, 'outputnode.func', model_wf, 'inputnode.func')
        wf.connect(prep_wf, 'outputnode.contrasts', model_wf, 'inputnode.contrasts')
        wf.connect(prep_wf, 'outputnode.mask', model_wf, 'inputnode.mask')
        
        # join before fixedfx
        joinfields = ["mask_files", "dof_file", "copes", "varcopes"]
        joiner = JoinNode(IdentityInterface(fields=joinfields),
                          joinsource='inputnode',
                          joinfield=joinfields,
                          name="joiner-fsl")
        wf.connect(model_wf, "outputnode.mni_mask", joiner, "mask_files")
        wf.connect(model_wf, "outputnode.dof_file", joiner, "dof_file")
        wf.connect(model_wf, "outputnode.copes", joiner, "copes")
        wf.connect(model_wf, "outputnode.varcopes", joiner, "varcopes")
        
        def join_copes(copes, varcopes):
            """
            Has to be flexible enough to handle multiple runs
            Length of copes/varcopes should equal number of conditions
            """
            # if sublists, multiple contrasts
            if all(isinstance(i, list) for i in copes):
                copes = [list(cope) for cope in zip(*copes)]
                varcopes = [list(varc) for varc in zip(*varcopes)]
            else:
                copes = [copes]
                varcopes = [varcopes]
            return copes, varcopes
        
        copesjoin = Node(Function(function=join_copes,
                                  output_names=["copes", "varcopes"]),
                         name='copesjoiner')
        wf.connect(joiner, "copes", copesjoin, "copes")
        wf.connect(joiner, "varcopes", copesjoin, "varcopes")
        
        fixed_fx = create_fixed_effects_flow()
        fixed_fx.get_node("l2model").inputs.num_copes = len(runs)
        
        # use the first mask since they should all be in same space
        wf.connect(joiner, ("mask_files", pickfirst), fixed_fx, "flameo.mask_file")
        wf.connect(joiner, "dof_file", fixed_fx, "inputspec.dof_files")
        wf.connect(copesjoin, "copes", fixed_fx, "inputspec.copes")
        wf.connect(copesjoin, "varcopes", fixed_fx, "inputspec.varcopes")
        
    def substitutes(contrasts):
        subs = []
        for i, con in enumerate(contrasts):
            name = con[0].replace(" ", "").replace(">", "_gt_").lower()
            # FSL subs
            subs.append(('_flameo%d/cope1.' % i, '%s_cope.' % name))
            subs.append(('_flameo%d/varcope1.' % i, '%s_varcope.' % name))
            subs.append(('_flameo%d/zstat1.' % i, '%s_zstat.' % name))
            subs.append(('_flameo%d/tstat1.' % i, '%s_tstat.' % name))
            # TODO: SPM subs
        return subs
    
    gensubs = Node(Function(function=substitutes), name="gensubs")
    wf.connect(prep_wf, "contrastgen.contrasts", gensubs, "contrasts")
            
    sinker = Node(DataSink(), name="datasink")
    sinker.inputs.base_directory = outdir
    
    wf.connect(gensubs, "out", sinker, "substitutions")
    if model == 'spm':
        wf.connect(model_wf, 'outputnode.spmT_images', sinker, 'mni.@T_images')
        wf.connect(model_wf, 'outputnode.con_images', sinker, 'mni.@con_images')
        wf.connect(model_wf, 'outputnode.mat_file', sinker, '@mat_file')
    elif model == 'fsl':
        wf.connect(fixed_fx, "outputspec.zstats", sinker, "mni.@zstats")
        wf.connect(fixed_fx, "outputspec.copes", sinker, "mni.@copes")
        wf.connect(fixed_fx, "outputspec.tstats", sinker, "mni.@tstats")
        wf.connect(fixed_fx, "outputspec.varcopes", sinker, "mni.@varcopes")

    return wf


def get_parser():
    docstr = "Subject level fMRI analysis using fmriprep outputs"
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument("bids_dir",
                        help="Root BIDS directory")
    parser.add_argument("model", choices=('spm', 'fsl'),
                        help="Model type: either spm or fsl")
    parser.add_argument("-f", dest="fmriprep_dir", 
                        help="Output directory of fmriprep")
    parser.add_argument("-s", dest="subjects", nargs="*",
                        help="List of subjects to process (default: all)")
    parser.add_argument("-t", dest="tasks", nargs="*",
                        help="List of tasks to process (default: all)")
    parser.add_argument("-ss", dest="session",
                        help="Session to process (default: None)")
    parser.add_argument("-w", dest="workdir", default=os.getcwd(),
                        help="Working directory")
    parser.add_argument("-o", dest="outdir", default=os.getcwd(),
                        help="Output directory")
    parser.add_argument("--sparse", action="store_true",
                        help="Specify a sparse model")
    parser.add_argument("-p", dest="plugin",
                        help="Nipype plugin to use (default: MultiProc)")
    parser.add_argument("--drop", dest="drop", type=int, default=0,
                        help="Number of starting volumes (dummy scans) to remove")
    parser.add_argument("--smooth", dest='fwhm', type=float, nargs="?", const=6, default=None,
                        help="Smoothing kernel FWHM (mm) - skipped if not set")
    parser.add_argument("--highpass", type=float, nargs="?", const=128, default=0,
                        help="Highpass filter (secs) - skipped if not set")
    parser.add_argument("--lowpass", type=float, default=0,
                        help="Lowpass filter (secs) - skipped if not set")
    parser.add_argument("--regressors", nargs="*", 
                        default=['FramewiseDisplacement',
                                 'aCompCor00',
                                 'aCompCor01',
                                 'aCompCor02',
                                 'aCompCor03',
                                 'aCompCor04',
                                 'aCompCor05'],
                        help="fmriprep confounds regressors")
    return parser

def process_subject(layout, bids_dir, model, subj, task, fmriprep_dir, 
                    outdir, workdir, regressors, sparse, session, 
                    dropvols, fwhm, highpass, lowpass, TA=None):
    """Grab information and start nipype workflow
    We want to parallelize runs for greater efficiency
    """
    
    kwargs = dict(
        subject=subj,
        type='bold',
        task=task,
        extensions=["nii", "nii.gz"]
    )
    
    runs = [b.run for b in layout.get(**kwargs)]
    if not runs:
        print("No runs found for 'subject {} task {}'".format(subj, task))
        return

    kwargs.update(dict(return_type="file"))
    # assumes TR is same across runs
    epi = layout.get(**kwargs)[0]
    try:
        TR = layout.get_metadata(epi)["RepetitionTime"]
    except AttributeError:
        print("RepetitionTime is not defined for", epi)
        return
    
    kwargs.update(dict(type='events', extensions=['tsv']))
    events = layout.get(**kwargs)
    
    if sparse:
        try:
            TA = layout.get_metadata(epi)["AcquisitionTime"]
        except AttributeError:
            print("AcquisitionTime is not defined for", epi)
            

    # skip if events are not defined
    if not events:
        print("No event files found for ", subj)
        return

    outdir = op.join(outdir, 'sub-' + subj, task)
    
    wf = create_l1_wf(bids_dir, model, subj, task, fmriprep_dir, runs, 
                      regressors, outdir, events, session, TR, workdir, 
                      dropvols, fwhm, highpass, lowpass, TA=TA)
    return wf
    
def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)

    if not op.exists(args.bids_dir):
        raise IOError("BIDS directory {} not found.".format(args.bids_dir))

    fmriprep_dir = (args.fmriprep_dir if args.fmriprep_dir else 
                    op.join(args.bids_dir, 'derivatives', 'fmriprep'))

    if not op.exists(fmriprep_dir):
        raise IOError("fmriprep directory {} not found.".format(fmriprep_dir))

    # check if software is found
    if args.model == 'fsl':
        version = fsl.Info.version()
    elif args.model == 'spm':
        spm.SPMCommand.set_mlab_paths(paths=os.environ['SPM_PATH'])
        version = spm.Info.version()
    if version is None:
        print(args.model, "installation not found")
    print("Using", args.model, version)

    workdir, outdir = map(op.realpath, [args.workdir, args.outdir])
    if not op.exists(workdir):
        os.makedirs(workdir)

    layout = BIDSLayout(args.bids_dir, exclude=["derivatives"])

    tasks = (args.tasks if args.tasks else 
             [task for task in layout.get_tasks() if 'rest' not in task.lower()])
    subjects = args.subjects if args.subjects else layout.get_subjects()

    for subj in subjects:
        subj = subj[4:] if subj.startswith('sub-') else subj
        for task in tasks:
            wf = process_subject(layout, args.bids_dir, args.model, subj, task, 
                         
                                 fmriprep_dir, outdir, workdir, args.regressors,
                                 args.sparse, args.session, args.drop, args.fwhm, 
                                 args.highpass, args.lowpass)
            if not wf:
                print("Skipping {} {} - something is wrong".format(subj, task))
                continue

            wf.config["execution"]["crashfile_format"] = "txt"
            wf.config["execution"]["parameterize_dirs"] = False

            plugin = args.plugin if args.plugin else "MultiProc"
            wf.run(plugin=plugin)

if __name__ == "__main__":
    main()

