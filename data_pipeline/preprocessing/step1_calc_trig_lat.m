close all
clear all

%% directories and paths.
maindir = 'C:/projects/emo_EEG';
rawdir = ([maindir '/data/raw']);
outdir = ([maindir '/data/processed']);
stimdir = ([maindir '/stimuli']);
% Add function folder
addpath([maindir '/data_pipeline/preprocessing/functions']);
%Initialize EEGLAB
addpath ('C:/projects/eeglab2025.1.0');
eeglab nogui;

%% Sample rate used for future analysis
newsr = 128; %Hz

%% Sub selection
subs = {'sub-pilot_8'}; % subs that need to be processed e.g., 'sub-pilot_1'

% if no subjects defined, automatically detect folders
if isempty(subs)
    subfolders=dir(fullfile(rawdir,'sub*'));
    subfolders=subfolders([subfolders.isdir]); % keep only directories
    subs = {subfolders.name}; % keep subject names as strings
end
fprintf('\nSubjects to process:\n');
disp(subs);

%% Calculating trigger latency 
for sub = 1:length(subs)
    sub_indir = fullfile(rawdir, subs{sub});
    if ~exist(sub_indir, 'dir')
        error('Raw directory does not exist: %s', sub_indir);
    end
    sub_outdir = fullfile(outdir, subs{sub}); % Output directory for the subject
    if ~exist(sub_outdir, 'dir')
        mkdir(sub_outdir)
    end
    outcsv = fullfile(sub_outdir, sprintf('%s_emo_adjusted_triggertimes_%dHz.csv', ...
        subs{sub}, newsr));
    [~, fdone] = load_existing_csv(outcsv);
    sub_raw_files = find_subject_files(sub_indir, 'bdf', 'emo');
    
    for file = 1:length(sub_raw_files)
        bdf_file_path = sub_raw_files{file};
        [~, fname] = fileparts(bdf_file_path);
        if ismember(fname, fdone)
            fprintf('Skipping already processed file: %s\n', fname);
            continue
        end
        trigtimes = compute_trigger_lag_emo(bdf_file_path, stimdir, newsr, 1);
        save_trigger_csv(outcsv, trigtimes);
        close all
    end
end