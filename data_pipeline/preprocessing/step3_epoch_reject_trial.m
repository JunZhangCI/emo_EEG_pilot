close all
clear all

%% directories and paths.
maindir = 'C:/projects/emo_EEG';
datadir = ([maindir '/data/processed']);
% Add function folder
addpath([maindir '/data_pipeline/preprocessing/functions']);
%Initialize EEGLAB
addpath ('C:/projects/eeglab2025.1.0');
eeglab nogui;

%% Variables
epochmin = -2;
epochmax = 62;
do_ocular = 0;
do_baseline = 0;
do_reject = 0;
artifact_rmsstd_thre = [];
ica_done = false; % true: ICA first; false: ICA after

%% Sub selection
subs = {'sub-pilot_Jun'}; % subs that need to be processed e.g., 'sub-pilot_1'

% if no subjects defined, automatically detect folders
if isempty(subs)
    subfolders=dir(fullfile(datadir,'sub*'));
    subfolders=subfolders([subfolders.isdir]); % keep only directories
    subs = {subfolders.name}; % keep subject names as strings
end
fprintf('\nSubjects to process:\n');
disp(subs);

%% Loop over subject
for sub = 1:length(subs)
    subname = subs{sub};
    if ica_done
        sub_indir = fullfile(datadir, subname, 'ref_down_filt_chRej/ica');
    else
        sub_indir = fullfile(datadir, subname, 'ref_down_filt_chRej');
    end
    sub_outdir = fullfile(sub_indir, 'epoch_reject');
    if ~exist(sub_outdir,'dir')
        mkdir(sub_outdir);
    end
    logfile = fullfile(sub_outdir, [subname '_epoch_reject_log.csv']);
    if ~exist(logfile,'file')
        fid = fopen(logfile,'w');
        fclose(fid);
    end
    sub_in_files = find_subject_files(sub_indir, 'set', 'emo');
    for file = 1:length(sub_in_files)
        set_file_path = sub_in_files{file};
        [~, fname] = fileparts(set_file_path);
        outname = sprintf('%s_epoch_reject.set', fname);
        if exist(fullfile(sub_outdir, outname), 'file')
            fprintf('Skipping %s — already processed.\n', outname);
            continue
        end
        EEG = pop_loadset(set_file_path);
        [EEG, n_epoch_reject] = epoch_option_reject(EEG, epochmin, epochmax, ...
            do_baseline, do_reject, artifact_rmsstd_thre);
        EEG.preprocess{end+1} = 'epoch'; 
        if do_ocular 
            EEG = Ocular_EEG(EEG);
            EEG = remove_ch(EEG, {'HEOG','VE0G'}, 0);
            EEG.preprocess{end+1} = 'ocular';
            ocular_str = 'yes';
        else
            ocular_str = 'no';
        end
        if do_baseline
            do_baseline_ans = 'yes';
            EEG.preprocess{end+1} = 'baseline_correct';
        else
            do_baseline_ans = 'no';
        end
        if do_reject
            do_reject_ans = 'yes';
            EEG.preprocess{end+1} = 'trial_reject';
        else
            do_reject_ans = 'no';
        end
        date_str = string(datetime('now','Format','yyyy-MM-dd'));
        time_str = string(datetime('now','Format','HH:mm'));
        log_to_csv_oneRow(logfile, ...
            'File', outname, ...
            'Date', date_str, ...
            'Time', time_str, ...
            'do_ocular', ocular_str, ...
            'do_baseline', do_baseline_ans, ...
            'do_reject', do_reject_ans, ...
            'n_epoch_reject', n_epoch_reject)
        save_processed_eeg(EEG, outname, sub_outdir);
        close all
    end
end