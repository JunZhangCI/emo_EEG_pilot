%%
close all
clear all
tic
% Set paths
maindir = 'C:/projects/emo_EEG';
datadir = fullfile(maindir, 'data/processed');

% Find folder corresponding to targeted step
step = 4;
epoch_first = 1;
do_filter = false;
lp = 1;
hp = 15;

% Initialize EEGLAB package
addpath ('C:/projects/eeglab2025.1.0');
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab('nogui');

% subject selection
subs = {'sub-pilot_Jun'}; % subs that need to be processed e.g., 'sub-pilot_Jun'
if isempty(subs)
    subfolders=dir(fullfile(datadir,'sub*'));
    subfolders=subfolders([subfolders.isdir]); % keep only directories
    subs = {subfolders.name}; % keep subject names as strings
end
fprintf('Subjects to process:\n');
disp(subs);

% loop over subjects
for sub = 1:length(subs)
    % Load all .set files in specific subject folder
    subname = subs{sub}; % e.g., 'sub-pilot_Jun'
    if step == 2
        sub_path = fullfile(datadir, subname, 'ref_down_filt_chRej');
    elseif step == 3
        if epoch_first
            sub_path = fullfile(datadir, subname, 'ref_down_filt_chRej/epoch_reject');
        else
            sub_path = fullfile(datadir, subname, 'ref_down_filt_chRej/ica');
        end
    elseif step == 4
        if epoch_first
            sub_path = fullfile(datadir, subname, 'ref_down_filt_chRej/epoch_reject/ica');
        else
            sub_path = fullfile(datadir, subname, 'ref_down_filt_chRej/ica/epoch_reject');
        end
    end   
    disp(sub_path);
    % Find all .set files in each subject's subfolder
    set_files = dir(fullfile(sub_path, '*.set'));
    if isempty (set_files)
        fprintf ('!Error!: No .set files found for %s, skipping.\n',subname);
        continue;
    end
    % Create output path
    out_path = fullfile(sub_path, "MAT");
    if ~exist(out_path, 'dir')
        mkdir (out_path)
    end 
    fprintf('\n---Processing %s ---\n', subname);

    % loop over each set files
    for j = 1:length(set_files)
        [~, base, ~] = fileparts(set_files(j).name);
        tmpname = [base '.mat'];
        outname = fullfile(out_path, tmpname);

        % Check whether there is existing processed file or not
        if exist(outname, 'file')
            fprintf('File already exists: %s. Skipping...\n', outname);
            continue; % Skip to the next file if the output file already exists
        end
        EEG = pop_loadset('filepath', sub_path, 'filename', set_files(j).name);
        if do_filter
            EEG = filter_EEG(EEG, hp, lp, 1);
        end
        EEG = eeg_checkset(EEG);
        save(outname, 'EEG', '-v7.3');

        % reset EEG variables
        ALLEEG = []; EEG = []; CURRENTSET = [];
    end
end
toc

