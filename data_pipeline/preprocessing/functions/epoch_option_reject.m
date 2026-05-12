function [EEG, n_epoch_reject] = epoch_option_reject(EEG, epochmin, epochmax, do_baseline, do_reject, artifact_rmsstd_thre)

if do_reject
    if ~exist('artifact_rmsstd_thre','var') || isempty(artifact_rmsstd_thre)
        error('Please set a std of RMS threshold for epoch rejection.');
    end
else
    artifact_rmsstd_thre = [];  % optional: make explicit it's unused
end

%% Epoching
% List of event types to epoch around
evt_types = unique(cellfun(@num2str, {EEG.event.type}, 'UniformOutput', false));
EEG = pop_epoch(EEG, evt_types, [epochmin epochmax]);
EEG = eeg_checkset(EEG);

% eeg_idx = find(strcmpi({EEG.chanlocs.type}, 'EEG'));
% EEG = pop_select(EEG, 'channel', eeg_idx);
EEG.data = double(EEG.data);
output_shape = size(EEG.data);  % [channels, samples, epochs]
fprintf('Output data shape: %d channels × %d samples × %d epochs\n', ...
        output_shape(1), output_shape(2), output_shape(3));

% Step 3: Baseline correct each epoch
if do_baseline
    baseline_window = [epochmin 0];   % ms
    EEG = pop_rmbase(EEG, baseline_window);
    disp('Baseline correction applied.');
end

%% Optional: Artifact rejection
if do_reject
    % 1) Average across channels -> [1 x nTime x nEpoch]
    epoch_avg_wave = squeeze(mean(EEG.data, 1));   % becomes [nTime x nEpoch]
    % 2) RMS for each epoch across time
    epoch_rms = sqrt(mean(epoch_avg_wave.^2, 1));  % [1 x nEpoch]
    % 3) Mean and std of RMS across epochs
    mu  = mean(epoch_rms);
    sig = std(epoch_rms);
    % 4) Reject epochs > mean + 2*std
    artifact_epoch_idx = find(epoch_rms > (mu + artifact_rmsstd_thre*sig));
    % Reject (no confirmation)
    EEG = pop_rejepoch(EEG, artifact_epoch_idx, 0);
    n_epoch_reject = numel(artifact_epoch_idx);
    fprintf('Rejected %d/%d epochs (threshold = %.4f).\n', ...
    n_epoch_reject, EEG.trials + n_epoch_reject, mu + 2*sig);
else
    n_epoch_reject = [];
end

EEG = eeg_checkset(EEG);

end