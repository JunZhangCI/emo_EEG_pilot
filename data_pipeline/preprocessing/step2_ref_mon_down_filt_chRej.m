close all
clear all

%% directories and paths.
maindir = 'C:/projects/emo_EEG';
rawdir = ([maindir '/data/raw']);
outdir = ([maindir '/data/processed']);
% Add function folder
addpath([maindir '/data_pipeline/preprocessing/functions']);
%Initialize EEGLAB
addpath ('C:/projects/eeglab2025.1.0');
eeglab nogui;

%% Variables
newsr = 128; % downsample rate will be used in following steps
% rawsr = 16384;
ref_ch = {'EXG5'}; % mastoid: M1 & M2; Average: {}
disact_ch_CI = {}; % For CI participant, state the disactived EEG channels due to contact with CI here
hp = 1;
lp = 30;
flat_thresh = 3e-6;

if isempty(ref_ch)
    ref_str = 'avg';
    unused_ch = {'EXG5', 'EXG6','EXG7','EXG8', 'M1', 'M2', 'Erg1'};
    montdir = select_montage(maindir, 'default');
elseif isequal(ref_ch, {'EXG5'})
    ref_str = strjoin(ref_ch, ''); 
    unused_ch = {'EXG6','EXG7','EXG8', 'M1', 'M2', 'Erg1'};
    montdir = select_montage(maindir, 'nose');
else
    ref_str = strjoin(ref_ch, '');  
    unused_ch = {'EXG5','EXG6','EXG7','EXG8', 'Erg1'}; 
    montdir = select_montage(maindir, 'default');
end

%% Sub selection
subs = {'sub-pilot_Jun'}; % subs that need to be processed e.g., 'sub-pilot_1'

% if no subjects defined, automatically detect folders
if isempty(subs)
    subfolders=dir(fullfile(rawdir,'sub*'));
    subfolders=subfolders([subfolders.isdir]); % keep only directories
    subs = {subfolders.name}; % keep subject names as strings
end
fprintf('\nSubjects to process:\n');
disp(subs);

%% Loop over subjects
for sub = 1:length(subs)
    subname = subs{sub};
    sub_rawdir = fullfile(rawdir, subname);
    if ~exist(sub_rawdir, 'dir')
        error('Raw directory does not exist: %s', sub_rawdir);
    end
    sub_outdir = fullfile(outdir, subname, 'ref_down_filt_chRej'); % Output directory for the subject
    if ~exist(sub_outdir, 'dir')
        mkdir(sub_outdir)
    end
    sub_raw_files = find_subject_files(sub_rawdir, 'bdf', 'emo');
    logfile = fullfile(sub_outdir, [subname '_chRej_log.csv']);
    if ~exist(logfile,'file')
        fid = fopen(logfile,'w');
        fclose(fid);
    end
    sub_lag_dir = fullfile(outdir, subname);

%% Loop over files
    for file = 1:length(sub_raw_files)
        bdf_file_path = sub_raw_files{file};
        [~, fname] = fileparts(bdf_file_path);
        outname = sprintf('%s_down%dHz_ref%s_filt%d-%dHz_chRej.set', fname, newsr, ...
            ref_str, hp, lp);
        if exist(fullfile(sub_outdir, outname), 'file')
            fprintf('Skipping %s — already processed.\n', outname);
            continue
        end
        fprintf('\n--- Processing %s ---\n', fname);

        % Process the file
        EEG = load_and_montage(bdf_file_path, montdir);
        EEG.preprocess = {};
        EEG.preprocess{end+1} = 'montage';
        EEG = assign_ch_type(EEG);
        EEG.preprocess{end+1} = 'assign_ch_type';
        EEG = remove_ch(EEG, unused_ch, 0);
        EEG.preprocess{end+1} = 'remove_unused_ch';
        [EEG, ref_str] = rereference_and_cleanup(EEG, ref_ch);       
        EEG.preprocess{end+1} = 'rereference';
        EEG = downsample(EEG, newsr);
        EEG.preprocess{end+1} = 'downsample';
        EEG = filter_EEG(EEG, hp, lp, 0);
        EEG.preprocess{end+1} = 'filter';
        [EEG, removed_str, flat_str] = manual_chRej(EEG, fname, flat_thresh);        
        EEG.preprocess{end+1} = 'manual_ch_rej';        
        EEG = update_triggers_from_csv(sub_lag_dir, fname, EEG);
        EEG.preprocess{end+1} = 'update_trigger';
        if ~isempty(disact_ch_CI)
            EEG = remove_ch(EEG, disact_ch_CI, 1);
            EEG.preprocess{end+1} = 'remove_CI_ch';
        end

        % save the processed file
        disact_str = strjoin(disact_ch_CI, ' ');  % for logging       
        save_processed_eeg(EEG, outname, sub_outdir);
        date_str = string(datetime('now','Format','yyyy-MM-dd'));
        time_str = string(datetime('now','Format','HH:mm'));
        
        log_to_csv_oneRow( logfile,...
            'File', outname, ...
            'Date', date_str, ...
            'Time', time_str, ...
            'Ref_ch', ref_str, ...
            'Disact_ch', disact_str, ...
            'Flat_ch', flat_str,...
            'Removed_ch', removed_str);
        close all
    end
end