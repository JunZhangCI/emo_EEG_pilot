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
focus_ch = 'AF3 AF4 Fz F3 F4 FC1 FC2';
epoch_done = 1; % true: ICA after; false: ICA first

%% Sub selection
subs = {'sub-pilot_Jun'}; % subs that need to be processed e.g., 'sub-pilot_1'
% subs = {'sub-pilot_Jun'};
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
    if epoch_done
        sub_indir = fullfile(datadir, subname, 'ref_down_filt_chRej/epoch_reject');
    else
        sub_indir = fullfile(datadir, subname, 'ref_down_filt_chRej');
    end
    sub_outdir = fullfile(sub_indir, 'ica');
    if ~exist(sub_outdir,'dir')
        mkdir(sub_outdir);
    end
    logfile = fullfile(sub_outdir, [subname '_ica_log.csv']);
    if ~exist(logfile,'file')
        fid = fopen(logfile,'w');
        fclose(fid);
    end
    sub_in_files = find_subject_files(sub_indir, 'set', 'emo');

    %% Loop over file
    for file = 1:length(sub_in_files)
        set_file_path = sub_in_files{file};
        [~, fname] = fileparts(set_file_path);
        outname = sprintf('%s_ica.set', fname);
        if exist(fullfile(sub_outdir, outname), 'file')
            fprintf('Skipping %s — already processed.\n', outname);
            continue
        end
        EEG = pop_loadset(set_file_path);
        [EEG_cleaned, rejectedStr, ~] = run_sobi_manual_reject_v2( ...
            EEG, focus_ch);
        EEG.preprocess{end+1} = 'ica';
        save_processed_eeg(EEG_cleaned, outname, sub_outdir)
        date_str = string(datetime('now','Format','yyyy-MM-dd'));
        time_str = string(datetime('now','Format','HH:mm'));
        log_to_csv_oneRow(logfile, ...
            'File', outname, ...
            'Date', date_str, ...
            'Time', time_str, ...
            'IC_reject1', rejectedStr)
        choice = questdlg( ...
        'Do you want to run ICA on this data again?', ...
        'Rerun ICA', ...
        'Yes','No','No');
        if strcmp(choice,'Yes')
            [EEG_recleaned, rejectedStr_reclean, ~] = run_sobi_manual_reject_v2( ...
            EEG_cleaned, focus_ch);
            [~, base, ~]= fileparts(outname);
            reclean_outname = sprintf('%s_ica2.set', base);
            EEG.preprocess{end+1} = 'ica2';
            save_processed_eeg(EEG_recleaned, reclean_outname, sub_outdir);
            log_to_csv_oneRow(logfile, ...
                'File', outname, ...
                'Date', date_str, ...
                'Time', time_str, ...
                'IC_reject2', rejectedStr_reclean);
        end
        close all
    end
end
