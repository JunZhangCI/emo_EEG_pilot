close all
clear all

%% directories and paths
maindir = 'C:/projects/emo_EEG';
datadir = [maindir '/data/processed'];
sent_audiodir = [maindir '/emo_audio/cleaned'];
corpusdir = [maindir '/emo_audio/mfa_corpus'];
emo_orderdir = [maindir '/stimuli/orders'];
concat_audiodir = [maindir '/emo_audio/random_cont'];
unique_phoneme_path = [corpusdir '/unique_phones.txt'];

addpath([maindir '/data_pipeline/PRP_extraction/functions']);

% EEGLAB
addpath('C:/projects/eeglab2025.1.0');
eeglab nogui;

%% settings
eeg_fs = 128;
pre_event_dur = 2;   % sec
sentence_isi = 0.5;  % sec

subject_ids = {'sub-pilot_1'};
subject_ids = {'sub-pilot_Jun'};

if isempty(subject_ids)
    subject_folders = dir(fullfile(datadir, 'sub*'));
    subject_folders = subject_folders([subject_folders.isdir]);
    subject_ids = {subject_folders.name};
end

% whitelist phonemes
phonemes = strsplit(strtrim(fileread(unique_phoneme_path)), newline);
phonemes = erase(phonemes, char(13));
phonemes = phonemes(:);   % nPhonemes x 1
nPhonemes = numel(phonemes);

fprintf('Whitelist phonemes (%d):\n', nPhonemes);
disp(phonemes)

%% loop over subjects
for sub_idx = 1:length(subject_ids)
    subject_id = subject_ids{sub_idx};
    fprintf('\n---- Processing: %s ----\n', subject_id);

    subject_input_dir = fullfile(datadir, subject_id, ...
        'ref_down_filt_chRej/epoch_reject');

    set_files = find_subject_files(subject_input_dir, 'set', 'emo');

    for file_idx = 1:length(set_files)
        set_file_path = set_files{file_idx};
        [set_folder, input_file_name, ext] = fileparts(set_file_path);

        fprintf('\n---- Loading: %s ----\n', [input_file_name ext]);

        %% load set
        EEG = pop_loadset('filename', [input_file_name ext], 'filepath', set_folder);

        if ndims(EEG.data) ~= 3
            error('Expected epoched EEG.data with size [channels x samples x epochs].');
        end

        nSamples = size(EEG.data, 2);
        nEpochs = size(EEG.data, 3);

        %% load matching order file
        order_num_token = regexp(input_file_name, 'acq-(\d+)', 'tokens', 'once');
        if isempty(order_num_token)
            fprintf('Could not parse acquisition number from %s\n', input_file_name);
            continue
        end

        order_num = str2double(order_num_token{1});
        order_file_path = fullfile(emo_orderdir, sprintf('order%d.xlsx', order_num));

        if ~exist(order_file_path, 'file')
            fprintf('Order file not found: %s\n', order_file_path);
            continue
        end

        block_order_table = readtable(order_file_path);
        block_audio_paths = block_order_table.audio_path;
        num_blocks = height(block_order_table);

        if num_blocks ~= nEpochs
            warning('Order file blocks (%d) do not match EEG epochs (%d). Using min.', ...
                num_blocks, nEpochs);
        end
        nUseEpochs = min(num_blocks, nEpochs);

        %% initialize new labels
        % size: phonemes x samples x epochs
        phoneme_onset_index = zeros(nPhonemes, nSamples, nUseEpochs, 'uint8');

        skipped_before_epoch = 0;
        skipped_after_epoch = 0;
        skipped_not_whitelist = 0;

        %% loop through epochs
        for epoch_idx = 1:nUseEpochs
            [~, block_audio_stem_raw, audio_ext] = fileparts(block_audio_paths{epoch_idx});
            block_audio_stem = erase(string(block_audio_stem_raw), "_scaled");
            block_audio_filename = string(block_audio_stem) + string(audio_ext);

            name_parts = split(string(block_audio_stem), "_");
            if numel(name_parts) < 3
                fprintf('Skipping malformed block name: %s\n', block_audio_stem);
                continue
            end

            current_gender = name_parts(1);
            current_speech_style = name_parts(2);

            sentence_order_dir = fullfile(concat_audiodir, current_speech_style, ...
                current_gender + "_" + current_speech_style);
            sentence_order_name = sprintf('%s_%s_random_cont_order.csv', ...
                current_gender, current_speech_style);
            sentence_order_path = fullfile(sentence_order_dir, sentence_order_name);

            if ~isfile(sentence_order_path)
                fprintf('Sentence order table not found: %s\n', sentence_order_name);
                continue
            end

            sentence_order_list = getSentenceOrderFromTable(sentence_order_path, block_audio_filename);

            sentence_start_time_epoch = pre_event_dur;

            for sentence_idx = 1:length(sentence_order_list)
                sentence_id = sentence_order_list{sentence_idx};

                sentence_textgrid_name = sprintf('%s_cleaned.TextGrid', sentence_id);
                sentence_textgrid_dir = fullfile(corpusdir, current_speech_style, ...
                    current_gender + "_" + current_speech_style, 'aligned');
                sentence_textgrid_path = fullfile(sentence_textgrid_dir, sentence_textgrid_name);

                sentence_audio_name = sprintf('%s_cleaned_scaled.wav', sentence_id);
                sentence_audio_dir = fullfile(sent_audiodir, current_speech_style, ...
                    current_gender + "_" + current_speech_style, 'scaled');
                sentence_audio_path = fullfile(sentence_audio_dir, sentence_audio_name);

                if ~isfile(sentence_textgrid_path)
                    fprintf('Missing TextGrid: %s\n', sentence_textgrid_path);
                    continue
                end

                if ~isfile(sentence_audio_path)
                    fprintf('Missing audio: %s\n', sentence_audio_path);
                    continue
                end

                [sentence_audio_data, sentence_audio_fs] = audioread(sentence_audio_path);
                sentence_audio_duration = length(sentence_audio_data) / sentence_audio_fs;

                phoneme_info = readPhonesFromTextGrid(sentence_textgrid_path);
                phoneme_onsets_sentence = phoneme_info.t1;
                phoneme_labels_sentence = phoneme_info.labels;

                sentence_duration = max(phoneme_info.tmax, sentence_audio_duration);

                for phoneme_idx = 1:length(phoneme_labels_sentence)
                    current_phoneme = string(phoneme_labels_sentence(phoneme_idx));

                    ph_idx = find(strcmp(phonemes, current_phoneme), 1);
                    if isempty(ph_idx)
                        skipped_not_whitelist = skipped_not_whitelist + 1;
                        continue
                    end

                    onset_time_epoch_sec = sentence_start_time_epoch + phoneme_onsets_sentence(phoneme_idx);
                    onset_sample_idx = round(onset_time_epoch_sec * eeg_fs) + 1;

                    if onset_sample_idx < 1
                        skipped_before_epoch = skipped_before_epoch + 1;
                        continue
                    elseif onset_sample_idx > nSamples
                        skipped_after_epoch = skipped_after_epoch + 1;
                        continue
                    end

                    phoneme_onset_index(ph_idx, onset_sample_idx, epoch_idx) = 1;
                end

                sentence_start_time_epoch = sentence_start_time_epoch + sentence_duration + sentence_isi;
            end

            fprintf('Epoch %d/%d done\n', epoch_idx, nUseEpochs);
        end

        %% add new labels to EEG struct
        EEG.phoneme_onset_index = phoneme_onset_index;  % [phonemes x samples x epochs]
        EEG.phonemes = phonemes;                        % [phonemes x 1]

        %% export updated set
        EEG = eeg_checkset(EEG);
        pop_saveset(EEG, 'filename', [input_file_name ext], 'filepath', set_folder);

        fprintf('Updated and saved: %s\n', fullfile(set_folder, [input_file_name ext]));
        fprintf('phoneme_onset_index size = [%d x %d x %d]\n', ...
            size(EEG.phoneme_onset_index, 1), ...
            size(EEG.phoneme_onset_index, 2), ...
            size(EEG.phoneme_onset_index, 3));
        fprintf('phonemes size = [%d x 1]\n', numel(EEG.phonemes));
    end
end