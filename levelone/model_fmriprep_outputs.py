"""
Individual and (group level?) analysis using `fmriprep` outputs and FSL.

Adapted script from original notebook:
https://bit.ly/2Scu6AT

Requirement: BIDS dataset (including events.tsv), 
fmriprep outputs, and modeling files [more on this when concrete]

TODO: use pybids derivatives fetcher
"""
from nipype.workflows.fmri.fsl import create_fixed_effects_flow
import nipype.algorithms.modelgen as model
from  nipype.interfaces import fsl, ants    
from nipype.interfaces.base import Bunch
from nipype import Workflow, Node, IdentityInterface, Function, DataSink, JoinNode
import os
import os.path as op
import argparse
from bids.grabbids import BIDSLayout

import numpy as np
from nibabel.testing import data_path
import nilearn.plotting
import matplotlib.pyplot as plt
import nibabel as nb


__version__ = '0.0.1'


def create_firstlevel_workflow(bids_dir, subj, task, fmriprep_dir, runs, outdir,
                               events, session, TR, sparse, workdir, dropvols,
                               fwhm, highpass, lowpass, name="sub-{}_task-{}_levelone"):
    """Processing pipeline"""

    # initialize workflow
    wf = Workflow(name=name.format(subj, task), 
                  base_dir=workdir)

    infosource = Node(IdentityInterface(fields=['run_id', 'event_file']), name='infosource')
    infosource.iterables = [('run_id', runs),
                            ('event_file', events)]
    infosource.synchronize = True


    def data_grabber(subj, task, fmriprep_dir, session, run_id):
        """Quick filegrabber ala SelectFiles/DataGrabber"""
        import os.path as op
        prefix = 'sub-{}_task-{}_run-{:01d}'.format(subj, task, run_id)
        fmriprep_func = op.join(fmriprep_dir, "sub-{}".format(subj), "func")
        if session:
            prefix = 'sub-{}_ses-{}_task-{}_run-{:01d}'.format(
                subj, session, task, run_id
            )
            fmriprep_func = op.join(fmriprep_dir, "sub-{}".format(subj), 
                                    "ses-{}".format(session), "func")

        # grab these files
        confound_file = op.join(fmriprep_func, "{}_bold_confounds.tsv".format(prefix))
        mni_file = op.join(
            fmriprep_func, 
            "{}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz".format(prefix)
        )
        mni_mask = mni_file.replace("_preproc.", "_brainmask.")
        return confound_file, mni_file, mni_mask

    datasource = Node(Function(output_names=["confound_file", 
                                             "mni_file", 
                                             "mni_mask"],
                               function=data_grabber), 
                      name='datasource')
    datasource.inputs.subj = subj
    datasource.inputs.task = task
    datasource.inputs.fmriprep_dir = fmriprep_dir
    datasource.inputs.session = session
    wf.connect(infosource, "run_id", datasource, "run_id")


    def gen_model_info(event_file, confound_file, regressor_names, dropvols):
        """Defines `SpecifyModel` information from BIDS events."""
        import pandas as pd
        from nipype.interfaces.base import Bunch
        
        events = pd.read_csv(event_file, sep="\t")
        trial_types = events.trial_type.unique()
        onset = []
        duration = []
        for trial in trial_types:
            onset.append(events[events.trial_type == trial].onset.tolist())
            duration.append(events[events.trial_type == trial].duration.tolist())
            
        confounds = pd.read_csv(confound_file, sep="\t", na_values="n/a")
        regressors = []
        for regressor in regressor_names:
            if regressor == 'FramewiseDisplacement':
                regressors.append(confounds[regressor].fillna(0)[dropvols:])
            else:
                regressors.append(confounds[regressor][dropvols:])


        info = [Bunch(
            conditions=trial_types,
            onsets=onset,
            durations=duration,
            regressors=regressors,
            regressor_names=regressor_names,
        )]
        return info

    modelinfo = Node(Function(function=gen_model_info), name="modelinfo")
    modelinfo.inputs.dropvols = dropvols
    
    # these will likely be in bids model json in the future
    modelinfo.inputs.regressor_names = [
        'FramewiseDisplacement',
        'aCompCor00',
        'aCompCor01',
        'aCompCor02',
        'aCompCor03',
        'aCompCor04',
        'aCompCor05',
    ]
    wf.connect(infosource, "event_file", modelinfo, "event_file")
    wf.connect(datasource, "confound_file", modelinfo, "confound_file")

    if dropvols:
        roi = Node(fsl.ExtractROI(t_min=dropvols, t_size=-1), name="extractroi")
        wf.connect(datasource, "mni_file", roi, "in_file")

    # smooth
    if fwhm:
        smooth = Node(fsl.IsotropicSmooth(), name='smooth')
        smooth.inputs.fwhm = fwhm
        if dropvols:
            wf.connect(roi, 'roi_file', smooth, 'in_file')
        else:
            wf.connect(datasource, 'mni_file', smooth, 'in_file')
    
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
        imgdata = img.get_data()
        vols = img.shape[-1]

        lidx = int(np.round((1. / lowpass) / fs * vols) if lowpass > 0
                   else vols // 2 + 1)
        hidx = int(np.round((1. / highpass) / fs * vols) if highpass > 0
                   else 0)

        F = np.zeros((vols))
        F[hidx:lidx] = 1
        F = ((F + F[::-1]) > 0).astype(int)
        if np.all(F == 1):
            filtered_data = imgdata
        else:
            filtered_data = np.real(np.fft.ifftn(np.fft.fftn(imgdata) * F))

        img_out = img.__class__(filtered_data, 
                                img.affine,
                                img.header)
        img_out.to_filename(out_file)
        return out_file

    if highpass or lowpass:
        bandpass = Node(Function(function=bandpass_filter), name='bandpass')
        bandpass.inputs.highpass = highpass
        bandpass.inputs.lowpass = lowpass
        bandpass.inputs.fs = 1./TR
        
        # TODO: generalize this
        if fwhm:
            wf.connect(smooth, 'out_file', bandpass, 'in_file')
        elif dropvols:
            wf.connect(roi, 'roi_file', bandpass, 'in_file')
        else:
            wf.connect(datasource, 'mni_file', bandpass, 'in_file')
        
    if sparse:
        modelspec = Node(model.SpecifySparseModel(), name="modelspec")
        modelspec.inputs.time_acquisition = None  # add option
    else:
        modelspec = Node(model.SpecifyModel(), name="modelspec")
    modelspec.inputs.input_units = "secs"
    modelspec.inputs.time_repetition = TR
    modelspec.inputs.high_pass_filter_cutoff = highpass
    wf.connect(modelinfo, "out", modelspec, "subject_info")
    
    if highpass:
        wf.connect(bandpass, 'out', modelspec, 'functional_runs')
    elif fwhm:
        wf.connect(smooth, 'out_file', modelspec, 'functional_runs')
    elif dropvols:
        wf.connect(roi, "roi_file", modelspec, "functional_runs")
    else:
        wf.connect(datasource, "mni_file", modelspec, "functional_runs")
    
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
    contrastgen.config['execution']['local_hash_check'] = False

    level1design = Node(fsl.Level1Design(), name="level1design")
    level1design.inputs.interscan_interval = TR
    level1design.inputs.bases = {"dgamma": {"derivs": True}}
    level1design.inputs.model_serial_correlations = True
    wf.connect(modelspec, "session_info", level1design, "session_info")
    wf.connect(contrastgen, "contrasts", level1design, "contrasts")

    modelgen = Node(fsl.FEATModel(), name="modelgen")
    wf.connect(level1design, "fsf_files", modelgen, "fsf_file")
    wf.connect(level1design, "ev_files", modelgen, "ev_files")

    masker = Node(fsl.ApplyMask(), name="masker")
    wf.connect(datasource, "mni_mask", masker, "mask_file")
    
    if dropvols:
        wf.connect(roi, "roi_file", masker, "in_file")
    else:
        wf.connect(datasource, "mni_file", masker, "in_file")

    glm = Node(fsl.FILMGLS(), name="filmgls")
    glm.inputs.autocorr_noestimate = True
    wf.connect(masker, "out_file", glm, "in_file")
    wf.connect(modelgen, "design_file", glm, "design_file")
    wf.connect(modelgen, "con_file", glm, "tcon_file")
    wf.connect(modelgen, "fcon_file", glm, "fcon_file")

    join_fields = ["mask_files", "dof_file", "copes", "varcopes"]
    joiner = JoinNode(IdentityInterface(fields=join_fields),
                      joinsource='infosource',
                      joinfield=join_fields,
                      name="joiner")
    
    wf.connect(datasource, "mni_mask", joiner, "mask_files")
    wf.connect(glm, "dof_file", joiner, "dof_file")
    wf.connect(glm, "copes", joiner, "copes")
    wf.connect(glm, "varcopes", joiner, "varcopes")
    
    def join_copes(copes, varcopes):
        """Has to be flexible enough to handle multiple runs

        Length of copes/varcopes should equal number of conditions
        """
        # if sublists, multiple contrasts
        if all(isinstance(i, list) for i in copes):
            copes = [list(cope) for cope in zip(*copes)]
            varcopes = [list(varc) for varc in zip(*varcopes)]
        else:
            copes = [copes]
            varcopes = [varcopes]
        print('copes: ', copes)
        print('varcopes: ', varcopes)

        return copes, varcopes

    copesjoin = Node(Function(function=join_copes,
                                output_names=["copes", "varcopes"]),
                       name='copesjoiner')
    wf.connect(joiner, "copes", copesjoin, "copes")
    wf.connect(joiner, "varcopes", copesjoin, "varcopes")
    
    # fixed_effects to combine stats across runs
    fixed_fx = create_fixed_effects_flow()
    fixed_fx.config['execution']['crashfile_format'] = 'txt'

    fixed_fx.get_node("l2model").inputs.num_copes = len(runs)
    
    # use the first mask since they should all be in same space
    pickfirst = lambda x: x[0] if not isinstance(x, str) else x

    wf.connect(joiner, ("mask_files", pickfirst), fixed_fx, "flameo.mask_file")
    wf.connect(joiner, "dof_file", fixed_fx, "inputspec.dof_files")
    wf.connect(copesjoin, "copes", fixed_fx, "inputspec.copes")
    wf.connect(copesjoin, "varcopes", fixed_fx, "inputspec.varcopes")
    
    def substitutes(contrasts):
        """Datasink output path substitutes"""
        subs = []
        for i, con in enumerate(contrasts):
            # replace annoying chars in filename
            name = con[0].replace(" ", "").replace(">", "_gt_").lower()
            
            subs.append(('_flameo%d/cope1.' % i, '%s_cope.' % name))
            subs.append(('_flameo%d/varcope1.' % i, '%s_varcope.' % name))
            subs.append(('_flameo%d/zstat1.' % i, '%s_zstat.' % name))
            subs.append(('_flameo%d/tstat1.' % i, '%s_tstat.' % name))
        return subs
            
            
    gensubs = Node(Function(function=substitutes), name="substitute-gen")
    wf.connect(contrastgen, "contrasts", gensubs, "contrasts")
            
    # stats should equal number of conditions...
    sinker = Node(DataSink(), name="datasink")
    sinker.inputs.base_directory = outdir
    
    wf.connect(gensubs, "out", sinker, "substitutions")
    wf.connect(fixed_fx, "outputspec.zstats", sinker, "stats.mni.@zstats")
    wf.connect(fixed_fx, "outputspec.copes", sinker, "stats.mni.@copes")
    wf.connect(fixed_fx, "outputspec.tstats", sinker, "stats.mni.@tstats")
    wf.connect(fixed_fx, "outputspec.varcopes", sinker, "stats.mni.@varcopes")
    
    return wf

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument("bids_dir",
                        help="Root BIDS directory")
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
    parser.add_argument("--drop", dest="drop", type=int,
                        help="Number of starting volumes (dummy scans) to remove")
    parser.add_argument("--smooth", dest='fwhm', type=float, nargs="?", const=6, default=None,
                        help="Smoothing kernel FWHM (mm) - skipped if not set")
    parser.add_argument("--highpass", type=float, nargs="?", const=128, default=0,
                        help="Highpass filter (secs) - skipped if not set")
    parser.add_argument("--lowpass", type=float, default=0,
                        help="Lowpass filter (secs) - skipped if not set")
    return parser

def process_subject(layout, bids_dir, subj, task, fmriprep_dir, 
                    session, outdir, workdir, sparse, dropvols,
                    fwhm, highpass, lowpass):
    """Grab information and start nipype workflow
    We want to parallelize runs for greater efficiency
    
    """
    
    runs = [b.run for b in layout.get(subject=subj,
                                      type="bold",
                                      task=task,
                                      extensions="nii.gz")]
    if not runs:
        print("No bold {} runs found for subject {}".format(task, subj))
        return

    events = layout.get(subject=subj, type="events", task=task, return_type="file")

    # assumes TR is same across runs
    try:
        epi = layout.get(subject=subj, type="bold", task=task, return_type="file")[0]
        try:
            TR = layout.get_metadata(epi)["RepetitionTime"]
        except Exception:
            print("Cannot find TR for", epi)

    # skip if events are not defined
        if not events:
            print("No event files found for ", subj)
            return

        suboutdir = op.join(outdir, 'sub-' + subj, task)
    
        wf = create_firstlevel_workflow(bids_dir, subj, task, fmriprep_dir, runs, 
                                        suboutdir, events, session, TR, sparse, workdir, 
                                        dropvols, fwhm, highpass, lowpass)
        return wf
    except Exception:
        print("Cannot find", task)

def main(argv=None):
    parser = argparser()
    args = parser.parse_args(argv)

    if not op.exists(args.bids_dir):
        raise IOError("BIDS directory {} not found.".format(args.bids_dir))

    fmriprep_dir = (args.fmriprep_dir if args.fmriprep_dir else 
                    op.join(args.bids_dir, 'derivatives', 'fmriprep'))

    if not op.exists(fmriprep_dir):
        raise IOError("fmriprep directory {} not found.".format(fmriprep_dir))

    workdir, outdir = op.realpath(args.workdir), op.realpath(args.outdir)
    if not op.exists(workdir):
        os.makedirs(workdir)

    layout = BIDSLayout(args.bids_dir, exclude=["derivatives"])

    tasks = (args.tasks if args.tasks else 
             [task for task in layout.get_tasks() if 'rest' not in task.lower()])
    subjects = args.subjects if args.subjects else layout.get_subjects()

    for subj in subjects:
        for task in tasks:
            wf = process_subject(layout, args.bids_dir, subj, task, fmriprep_dir, 
                                 args.session, outdir, workdir, args.sparse, args.drop,
                                 args.fwhm, args.highpass, args.lowpass)
            if not wf:
                print("Skipping {} {} - something is wrong".format(subj, task))
                continue

            wf.config["execution"]["crashfile_format"] = "txt"
            wf.config["execution"]["parameterize_dirs"] = False

            plugin = args.plugin if args.plugin else "MultiProc"
            wf.run(plugin=plugin)

if __name__ == "__main__":
    main()
