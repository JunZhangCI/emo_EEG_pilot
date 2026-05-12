%% directories and paths.
clear all
maindir = 'C:/projects/emo_EEG';
datadir = ([maindir '/data/processed']);
corpusdir = ([maindir '/emo_audio/mfa_corpus']);
concat_audiodir = ([maindir '/emo_audio/random_cont']);
unique_phoneme_path = ([corpusdir '/unique_phones.txt']);
stimuli_audiodir = ([maindir '/stimuli']);
emo_orderdir = ([stimuli_audiodir '/orders']);
outdir = ([maindir '/data_pipeline/mTRF/dataCND']);
if ~exist(outdir,'dir')
    mkdir(outdir); 
end

% Add function folder
addpath([maindir '/data_pipeline/PRP_extraction/functions']);

%% settings
n_order = 1;
fs = 128;
ISI = 0.5;

% feature switches
use_env = true;
use_acoustic_onset = true;
use_ph_onset = true;
use_ph_dur = false;
use_mfcc13 = false;

% MFCC setting
n_mfcc = 13;
mfcc_win_sec = 0.025;   % 25 ms window
mfcc_hop_sec = 0.010;   % 10 ms hop

%% load unique phonemes
phonemes_to_include = strsplit(strtrim(fileread(unique_phoneme_path)), newline);
phonemes_to_include = erase(phonemes_to_include, char(13));   % remove \r
fprintf('Phonemes to include in data file (%d):\n', length(phonemes_to_include));
disp(phonemes_to_include)

% map phoneme -> column index
n_unique_ph = numel(phonemes_to_include);
ph2idx = containers.Map(phonemes_to_include, 1:n_unique_ph);

%% find order and corresponding csv
feature_names = {'Speech Envelope Vector', 'Phoneme Onset Vector', 'Phonemes', 'MFCC13'};
order_file_path = fullfile(emo_orderdir, sprintf('order%d.xlsx', n_order));
if exist(order_file_path, 'file')
    emo_order_tab = readtable(order_file_path);
else
    error('Order file not found.\n');
end

cont_audio_paths = emo_order_tab.audio_path;
stim_idx = emo_order_tab.stim_idx;
n_blocks = height(emo_order_tab);

%% preallocate outputs
env_array       = cell(1, n_blocks);   % each cell: T x 1
acoustic_onset_array = cell(1, n_blocks);   % each cell: T x 1
ph_onset_array  = cell(1, n_blocks);   % each cell: T x 1
ph_dur_array    = cell(1, n_blocks);   % each cell: T x n_unique_ph
mfcc_array      = cell(1, n_blocks);   % each cell: T x 13

emotions = cell(1, n_blocks);
speech_styles = cell(1, n_blocks);
genders = cell(1, n_blocks);
cont_audio_filenames = cell(1, n_blocks);

%% loop over each block
for block_idx = 1:n_blocks
    % get base name and extension
    [~, cont_name_orig, ext] = fileparts(cont_audio_paths{block_idx});
    cont_name = erase(string(cont_name_orig), "_scaled");
    cont_audio_fullname = string(cont_name) + string(ext); 
    cont_audio_fullname_orig = string(cont_name_orig) + string(ext);
    cont_audio_filenames{1, block_idx} = cont_audio_fullname_orig;
    
    % parse filename parts (assumes: gender_speechStyle_emotion_...)
    namepts = split(string(cont_name), "_");
    current_gender = char(namepts(1));
    current_speech_style = char(namepts(2));
    current_emotion = char(namepts(3));

    % save extra conditions
    emotions{1, block_idx} = current_emotion;
    speech_styles{1, block_idx} = current_speech_style;
    genders{1, block_idx} = current_gender;

    % load concatenated audio
    audio_file_path = fullfile(stimuli_audiodir, char(cont_audio_fullname_orig));
    [audio_data, fs_audio] = audioread(audio_file_path);

    % force mono if stereo
    if size(audio_data, 2) > 1
        audio_data = mean(audio_data, 2);
    end
    
    %% extract envelope
    analytical_signal = hilbert(audio_data);
    env = abs(analytical_signal);
    env_ds = resample(env, fs, fs_audio);
    env_ds = env_ds(:);   % force column vector

    %% derive acoustic onset from envelope
    % Eelbrain-like logic: positive first derivative of the envelope
    acoustic_onset_ds = [0; diff(env_ds)];
    acoustic_onset_ds(acoustic_onset_ds < 0) = 0;

    %% initialize outputs at same final length
    T = length(env_ds);
    ph_onset_vec = zeros(T, 1);
    ph_dur_mat = zeros(T, n_unique_ph);
    mfcc_mat_ds = zeros(T, n_mfcc);

    %% extract MFCCs and interpolate to T x 13 at fs = 128
    if use_mfcc13
        mfcc_win = hamming(round(mfcc_win_sec * fs_audio), 'periodic');
        mfcc_overlap = round((mfcc_win_sec - mfcc_hop_sec) * fs_audio);

        % coeffs: nFrames x 13
        % loc: sample index of each frame
        [coeffs, ~, ~, loc] = mfcc(audio_data, fs_audio, ...
            'Window', mfcc_win, ...
            'OverlapLength', mfcc_overlap, ...
            'NumCoeffs', n_mfcc, ...
            'LogEnergy', 'ignore');

        % approximate frame-center times in seconds
        frame_t = (loc - floor(numel(mfcc_win)/2) - 1) / fs_audio;

        % target time axis at final sampling rate
        target_t = (0:T-1)' / fs;

        % interpolate frame-wise MFCCs to sample-wise feature matrix
        mfcc_mat_ds = interp1(frame_t, coeffs, target_t, 'linear', 'extrap');

        % fill any boundary NaNs just in case
        if any(isnan(mfcc_mat_ds), 'all')
            mfcc_mat_ds = fillmissing(mfcc_mat_ds, 'nearest');
        end
    end

    %% build order-table path
    sent_order_tab_folder = fullfile(concat_audiodir, current_speech_style, current_gender + "_" + current_speech_style);
    sent_order_tab_name = sprintf('%s_%s_random_cont_order.csv', current_gender, current_speech_style);
    sent_order_tab_path = fullfile(sent_order_tab_folder, sent_order_tab_name);

    % load sentence order for current block
    if isfile(sent_order_tab_path)
        sent_order_list = getSentenceOrderFromTable(sent_order_tab_path, cont_audio_fullname);
    else
        fprintf('Sentence order table not found for %s.\n', sent_order_tab_name);
        continue;
    end

    %% loop over each sentence
    start_t = 0;
    for sentence_idx = 1:length(sent_order_list)
        % build sentence TextGrid path
        sent_txt_name = sprintf('%s_cleaned.TextGrid', sent_order_list{sentence_idx});
        sent_txt_folder = fullfile( ...
                                    corpusdir, ...
                                    current_speech_style, ...
                                    current_gender + "_" + current_speech_style, ...
                                    'aligned');
        sent_txt_path = fullfile(sent_txt_folder, sent_txt_name);

        % load intervals from TextGrid
        phoneme_info = readPhonesFromTextGrid(sent_txt_path);
        t1 = phoneme_info.t1;
        t2 = phoneme_info.t2;
        labels = phoneme_info.labels;
        end_t = phoneme_info.tmax;
        
        % extract phoneme onsets and durations
        for phoneme_idx = 1:length(labels)

            phoneme_label = char(labels{phoneme_idx});

            % skip phoneme not in included list
            if ~isKey(ph2idx, phoneme_label)
                continue
            end

            % absolute times in concatenated audio
            onset_t = t1(phoneme_idx) + start_t;
            offset_t = t2(phoneme_idx) + start_t;

            % convert to sample indices at target fs
            onset_idx = round(onset_t * fs) + 1;
            offset_idx = round(offset_t * fs) + 1;

            % keep within bounds
            onset_idx = max(1, min(T, onset_idx));
            offset_idx = max(1, min(T, offset_idx));

            if offset_idx < onset_idx
                continue
            end
            
            % overall phoneme onset vector
            ph_onset_vec(onset_idx) = 1;

            % phoneme duration matrix
            ph_col = ph2idx(phoneme_label);
            ph_dur_mat(onset_idx:offset_idx, ph_col) = 1; 
        end

        start_t = start_t + end_t + ISI;
    end 

    %% save current block outputs
    env_array{block_idx} = env_ds;
    acoustic_onset_array{block_idx} = acoustic_onset_ds;
    ph_onset_array{block_idx} = ph_onset_vec;
    ph_dur_array{block_idx} = ph_dur_mat;
    mfcc_array{block_idx} = mfcc_mat_ds;
end

%% sort according to stim_idx
valid_mask = ~isnan(stim_idx) & stim_idx > 0;
if ~all(valid_mask)
    error('Some stim_idx values are invalid.');
end

[stim_idx_sorted, sort_order] = sort(stim_idx); %#ok<ASGLU>

env_array = env_array(sort_order);
acoustic_onset_array = acoustic_onset_array(sort_order);
ph_onset_array = ph_onset_array(sort_order);
ph_dur_array = ph_dur_array(sort_order);
mfcc_array = mfcc_array(sort_order);

emotions = emotions(sort_order);
speech_styles = speech_styles(sort_order);
genders = genders(sort_order);
cont_audio_filenames = cont_audio_filenames(sort_order);

%% save feature struct
stim = struct();
stim.fs = fs;
stim.filenames = cont_audio_filenames;
stim.unique_ph = phonemes_to_include;
stim.emotions = emotions;
stim.speech_styles = speech_styles;
stim.genders = genders;
stim.ISI = ISI;

% save MFCC settings for reproducibility
stim.mfcc_settings = struct( ...
    'enabled', use_mfcc13, ...
    'num_coeffs', n_mfcc, ...
    'window_sec', mfcc_win_sec, ...
    'hop_sec', mfcc_hop_sec);

selected_data = {};
selected_names = {};

if use_env
    selected_data(end+1, :) = env_array;
    selected_names{end+1} = 'Speech Envelope Vector';
end
if use_acoustic_onset
    selected_data(end+1, :) = acoustic_onset_array;
    selected_names{end+1} = 'Acoustic Onset Vector';
end
if use_ph_onset
    selected_data(end+1, :) = ph_onset_array;
    selected_names{end+1} = 'Phoneme Onset Vector';
end
if use_ph_dur
    selected_data(end+1, :) = ph_dur_array;
    selected_names{end+1} = 'Phonemes';
end
if use_mfcc13
    selected_data(end+1, :) = mfcc_array;
    selected_names{end+1} = 'MFCC13';
end

if isempty(selected_names)
    error('No features selected. Turn on at least one feature switch.');
end

stim.data = selected_data;
stim.names = selected_names;

out_stim_path = fullfile(outdir, sprintf('dataStim_order%d.mat', n_order));
save(out_stim_path, 'stim');
% V1
% %% directories and paths.
% maindir = 'C:/projects/emo_EEG';
% datadir = ([maindir '/data/processed']);
% corpusdir = ([maindir '/emo_audio/mfa_corpus']);
% concat_audiodir = ([maindir '/emo_audio/random_cont']);
% unique_phoneme_path = ([corpusdir '/unique_phones.txt']);
% stimuli_audiodir = ([maindir '/stimuli']);
% emo_orderdir = ([stimuli_audiodir '/orders']);
% outdir = ([maindir '/data_pipeline/mTRF/dataCND']);
% if ~exist(outdir,'dir')
%     mkdir(outdir); 
% end
% 
% % Add function folder
% addpath([maindir '/data_pipeline/PRP_extraction/functions']);
% 
% % settings
% n_order = 1;
% fs = 128;
% ISI = 0.5;
% use_env      = true;
% use_ph_onset = true;
% use_ph_dur   = false;
% 
% % load unique phonemes
% phonemes_to_include = strsplit(strtrim(fileread(unique_phoneme_path)), newline);
% phonemes_to_include = erase(phonemes_to_include, char(13));   % remove \r
% fprintf('Phonemes to include in data file (%d):\n', length(phonemes_to_include));
% disp(phonemes_to_include)
% 
% % map phoneme -> column index
% n_unique_ph = numel(phonemes_to_include);
% ph2idx = containers.Map(phonemes_to_include, 1:n_unique_ph);
% 
% % find order and corresponding csv
% feature_names = {'Speech Envelope Vector', 'Phoneme Onset Vector', 'Phonemes'};
% n_features = numel(feature_names);
% order_file_path = fullfile(emo_orderdir, sprintf('order%d.xlsx', n_order));
% if exist(order_file_path, 'file')
%     emo_order_tab = readtable(order_file_path);
% else
%     error('Order file not found.\n');
% end
% cont_audio_paths = emo_order_tab.audio_path;
% stim_idx = emo_order_tab.stim_idx;
% n_blocks = height(emo_order_tab);
% 
% 
% %% preallocate outputs
% env_array       = cell(1, n_blocks);   % each cell: T x 1
% ph_onset_array  = cell(1, n_blocks);   % each cell: T x 1
% ph_dur_array = cell(1, n_blocks);   % each cell: T x n_unique_ph
% 
% emotions = cell(1, n_blocks);
% speech_styles = cell(1, n_blocks);
% genders = cell(1, n_blocks);
% cont_audio_filenames = cell(1, n_blocks);
% 
%  %% loop over each block
% for block_idx = 1:n_blocks
%     % get base name and extension
%     [~, cont_name_orig, ext] = fileparts(cont_audio_paths{block_idx});
%     cont_name = erase(string(cont_name_orig), "_scaled");
%     cont_audio_fullname = string(cont_name) + string(ext); 
%     cont_audio_fullname_orig = string(cont_name_orig) + string(ext);   % e.g., "Fem_CDS_xxx.wav"
%     cont_audio_filenames{1, block_idx} = cont_audio_fullname_orig;
% 
%     % parse filename parts (assumes: gender_speechStyle_...)
%     namepts = split(string(cont_name), "_");
%     current_gender = char(namepts(1));
%     current_speech_style = char(namepts(2));
%     current_emotion = char(namepts(3));
% 
%     % save the extra conditions (save as separate files)
%     emotions{1, block_idx} = current_emotion;
%     speech_styles{1, block_idx} = current_speech_style;
%     genders{1, block_idx} = current_gender;
% 
%     % load the concatenated audio and measure the duration
%     audio_file_path = fullfile(stimuli_audiodir, char(cont_audio_fullname_orig));
%     [audio_data, fs_audio] = audioread(audio_file_path);
% 
%     % extract grand amplitude envelope (Env)
%     analytical_signal = hilbert(audio_data);
%     env = abs(analytical_signal);
%     env_ds = resample(env, fs, fs_audio);
%     env_ds = env_ds(:);   % force column vector
% 
%     %% initialize phoneme outputs at same length as envelope
%     T = length(env_ds);
%     ph_onset_vec = zeros(T, 1);
%     ph_dur_mat = zeros(T, n_unique_ph);
% 
%     % build order-table path
%     sent_order_tab_folder = fullfile(concat_audiodir, current_speech_style, current_gender + "_" + current_speech_style);
%     sent_order_tab_name = sprintf('%s_%s_random_cont_order.csv', current_gender, current_speech_style);
%     sent_order_tab_path = fullfile(sent_order_tab_folder, sent_order_tab_name);
% 
%     % load the sentence order for the current block
%     if isfile(sent_order_tab_path)
%         sent_order_list = getSentenceOrderFromTable(sent_order_tab_path, cont_audio_fullname);
%     else
%         fprintf('Sentence order table not found for %s.\n', sent_order_tab_name);
%         continue;
%     end
% 
%     % loop over each sentence
%     start_t = 0;
%     for sentence_idx = 1:length(sent_order_list)
%         % build sentence textgrid path
%         sent_txt_name = sprintf('%s_cleaned.TextGrid', sent_order_list{sentence_idx});
%         sent_txt_folder = fullfile( ...
%                                     corpusdir, ...
%                                     current_speech_style, ...
%                                     current_gender + "_" + current_speech_style, ...
%                                     'aligned');
%         sent_txt_path = fullfile(sent_txt_folder, sent_txt_name);
% 
%         % load intervals from textgrid
%         phoneme_info = readPhonesFromTextGrid(sent_txt_path);
%         t1 = phoneme_info.t1;
%         t2 = phoneme_info.t2;
%         labels = phoneme_info.labels;
%         end_t = phoneme_info.tmax;
% 
%         % extract phoneme onsets                
%         for phoneme_idx = 1:length(labels)
% 
%             phoneme_label = char(labels{phoneme_idx});
%             % skip phoneme not in included list
%             if ~isKey(ph2idx, phoneme_label)
%                 continue
%             end
% 
%             % absolute times in concatenated audio
%             onset_t = t1(phoneme_idx) + start_t;
%             offset_t = t2(phoneme_idx) + start_t;
% 
%             % convert to sample indices at target fs
%             onset_idx = round(onset_t * fs) + 1;
%             offset_idx = round(offset_t * fs) + 1;
% 
%             % keep within bounds
%             onset_idx = max(1, min(T, onset_idx));
%             offset_idx = max(1, min(T, offset_idx));
% 
%             if offset_idx < onset_idx
%                 continue
%             end
% 
%             % binary overall phoneme onset vector
%             ph_onset_vec(onset_idx) = 1;
% 
%             % phoneme duration matrix
%             ph_col = ph2idx(phoneme_label);
%             ph_dur_mat(onset_idx:offset_idx, ph_col) = 1; 
%         end
%         start_t = start_t + end_t + ISI;
%     end 
%     %% save current block outputs
%     env_array{block_idx} = env_ds;
%     ph_onset_array{block_idx} = ph_onset_vec;
%     ph_dur_array{block_idx} = ph_dur_mat;
% end
% %% sort according to stim_idx
% valid_mask = ~isnan(stim_idx) & stim_idx > 0;
% if ~all(valid_mask)
%     error('Some stim_idx values are invalid.');
% end
% 
% [stim_idx_sorted, sort_order] = sort(stim_idx);
% 
% env_array = env_array(sort_order);
% ph_onset_array = ph_onset_array(sort_order);
% ph_dur_array = ph_dur_array(sort_order);
% 
% emotions = emotions(sort_order);
% speech_styles = speech_styles(sort_order);
% genders = genders(sort_order);
% cont_audio_filenames = cont_audio_filenames(sort_order);
% 
% %% save feature struct
% stim = struct();
% stim.fs = fs;
% stim.filenames = cont_audio_filenames;
% stim.unique_ph = phonemes_to_include;
% stim.emotions = emotions;
% stim.speech_styles = speech_styles;
% stim.genders = genders;
% stim.ISI = ISI;
% 
% selected_data = {};
% selected_names = {};
% 
% if use_env
%     selected_data(end+1, :) = env_array;
%     selected_names{end+1} = 'Speech Envelope Vector';
% end
% if use_ph_onset
%     selected_data(end+1, :) = ph_onset_array;
%     selected_names{end+1} = 'Phoneme Onset Vector';
% end
% if use_ph_dur
%     selected_data(end+1, :) = ph_dur_array;
%     selected_names{end+1} = 'Phonemes';
% end
% 
% if isempty(selected_names)
%     error('No features selected. Turn on at least one feature switch.');
% end
% 
% stim.data = selected_data;
% stim.names = selected_names;
% 
% out_stim_path = fullfile(outdir, sprintf('dataStim_order%d.mat', n_order));
% save(out_stim_path, 'stim');