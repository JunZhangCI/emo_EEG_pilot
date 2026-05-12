%% directories and paths.
maindir = 'C:/projects/emo_EEG';
datadir = ([maindir '/data/processed']);
concat_audiodir = ([maindir '/stimuli']);
emo_orderdir = ([concat_audiodir '/orders']);
outdir = ([maindir '/data_pipeline/mTRF/dataCND']);
if ~exist(outdir,'dir')
    mkdir(outdir); 
end

% Add function folder
addpath([maindir '/data_pipeline/PRP_extraction/functions']);

%% Variables
epoch_done = 1;

%% Sub selection
subs = {'sub-pilot_1', 'sub-pilot_2', 'sub-pilot_3', 'sub-pilot_4', 'sub-pilot_5'}; % subs that need to be processed e.g., 'sub-pilot_1'
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
    fprintf('\n---- Processing: %s ----\n', string(subname));
    if epoch_done
        sub_indir = fullfile(datadir, subname, 'ref_down_filt_chRej_ocular/epoch_reject/ica/MAT');
    else
        sub_indir = fullfile(datadir, subname, 'ref_down_filt_chRej_ocular/ica/epoch_reject/MAT');
    end

    sub_in_files = find_subject_files(sub_indir, 'mat', 'emo');
    %% loop over over each mat file
    for file = 1:length(sub_in_files)
        mat_file_path = sub_in_files{file};
        [~, fname] = fileparts(mat_file_path);
        parts = split(fname, "_");
        sub_idx = str2double(parts(2));
        ica_times = char(parts(end));
        outname = sprintf('pre_dataSub%d_%s.mat', sub_idx, ica_times);
        sub_out_path = fullfile(outdir, outname);

        % Load the data file
        mat_struct = load(mat_file_path); % 1x1 struct

        % extract EEG data
        EEG_struct = mat_struct.EEG;      % 1x1 EEGLAB struct
        EEG = EEG_struct.data;            % numeric array: chans × time × epochs (usually)
        reRef = EEG_struct.ref;
        fs = EEG_struct.srate;
        times = EEG_struct.times;
        chanlocs = permute(EEG_struct.chanlocs, [2, 1]); % chans x 1 -> 1 x chans
        
        % find order and corresponding csv
        n_order = str2double(regexp(fname,'acq-(\d+)','tokens','once'));
        order_file_path = fullfile(emo_orderdir, sprintf('order%d.xlsx', n_order));
        if exist(order_file_path, 'file')
            emo_order_tab = readtable(order_file_path);
        else
            fprintf('Order file not found for %s.\n', fname);
            continue;
        end
        cont_audio_paths = emo_order_tab.audio_path;
        stim_idx = emo_order_tab.stim_idx;
        n_blocks = height(emo_order_tab);

        % sanity check
        if size(EEG,3) ~= n_blocks
            error('Number of EEG epochs (%d) does not match number of rows in order file (%d) for %s.', ...
                size(EEG,3), n_blocks, fname);
        end
        
        % preallocate
        trial_eeg = cell(1, n_blocks);
        cont_audio_filenames = cell(1, n_blocks);

         %% loop over each block
        for block_idx = 1:n_blocks
            % get base name and extension
            [~, cont_name, ext] = fileparts(cont_audio_paths{block_idx});
            cont_audio_fullname = string(cont_name) + string(ext);   % e.g., "Fem_CDS_xxx.wav"
            cont_audio_filenames{1, block_idx} = cont_audio_fullname;

            % load the concatenated audio and measure the duration
            audio_file_path = fullfile(concat_audiodir, char(cont_audio_fullname));
            [audio_data, fs_audio] = audioread(audio_file_path);
            audio_duration = (length(audio_data) / fs_audio) *1000; % duration in miliseconds

            % load corresponding epoch of EEG data
            current_epoch_EEG = EEG(:,:,block_idx);   % chans × timepoints

            % reshape the data
            current_epoch_EEG = permute(current_epoch_EEG, [2, 1]); % timepoints x chans

            % trim the eeg 
            % use times to find sample idx in eegs curresponds to 0ms in
            % times and trim eeg data between 0 to (audio_duration)ms         
            start_idx = find(times >= 0, 1, 'first');
            end_idx = find(times <= audio_duration, 1, 'last');
            if isempty(start_idx)
                error('No sample at or after 0 ms for %s block %d.', fname, block_idx);
            end
            if isempty(end_idx)
                error('No sample at or before audio duration (%.2f ms) for %s block %d.', ...
                    audio_duration, fname, block_idx);
            end
            if end_idx < start_idx
                error('Invalid trim indices for %s block %d.', fname, block_idx);
            end
            current_epoch_EEG = current_epoch_EEG(start_idx:end_idx, :);

            % save processed current_epoch_EEG data to trial_eeg{block_idx}
            trial_eeg{1, block_idx} = current_epoch_EEG;
        end
        % basing on the index saved in the stim_idx, rearrange trial_eeg,
        % emotions, speech_styles and genders        
        % remove invalid stim_idx if any
        valid_mask = ~isnan(stim_idx) & stim_idx > 0;
        if ~all(valid_mask)
            error('Some stim_idx values are invalid in %s.', fname);
        end

        % sort according to stim_idx
        [~, sort_order] = sort(stim_idx);
        trial_eeg = trial_eeg(sort_order);

        % save output struct
        eeg = struct();
        eeg.dataType = 'EEG';
        eeg.deviceName = 'BioSemi ActiveTwo';
        eeg.data = trial_eeg;
        eeg.fs = fs;
        eeg.chanlocs = chanlocs;
        eeg.origTrialPosition = reshape(stim_idx, 1, []);
        eeg.originalFs = 16384;
        eeg.reRef = reRef;

        % save eeg structure
        save(sub_out_path, 'eeg', '-v7.3');
    end    
end

