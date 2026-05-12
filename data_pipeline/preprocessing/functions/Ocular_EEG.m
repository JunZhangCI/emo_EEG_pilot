function data_exported = Ocular_EEG(eeg_to_be_corrected)

% EEG = eeg_to_be_corrected.data_exported;
EEG = eeg_to_be_corrected;

% Find EOG automatically
labels = {EEG.chanlocs.labels};
[veog_idx, heog_idx] = find_eog_channels(labels);
eeg_idx = setdiff(1:size(EEG.data,1), [veog_idx heog_idx]);

% Handle 2D vs 3D data
orig_size = size(EEG.data);

if ndims(EEG.data) == 3
    % channels x timepoints x epochs  -->  channels x (timepoints*epochs)
    data_2d = reshape(EEG.data, orig_size(1), []);
    is_epoched = true;
else
    % already continuous: channels x timepoints
    data_2d = EEG.data;
    is_epoched = false;
end

VEOG = data_2d(veog_idx,:);
HEOG = data_2d(heog_idx,:);

% %
% eeg_to_be_corrected_data = eeg_to_be_corrected.data_exported.eeg_data;
% vertical_eye_selected = eeg_to_be_corrected.data_exported.eeg_data(36,:);
% horizontal_eye_selected = eeg_to_be_corrected.data_exported.eeg_data(35,:);
% 
% %
% data_selected_weights = eeg_to_be_corrected.data_exported.eeg_data;
% vertical_eye_selected_weights = data_selected_weights(36,:);
% horizontal_eye_selected_weights = data_selected_weights(35,:);

% eeg_to_be_corrected_data = EEG.data;
eeg_to_be_corrected_data = data_2d(eeg_idx,:);
data_selected_weights = data_2d(eeg_idx,:);

vertical_eye_selected = VEOG;
horizontal_eye_selected = HEOG;
% data_selected_weights = EEG.data;
vertical_eye_selected_weights = vertical_eye_selected;
horizontal_eye_selected_weights = horizontal_eye_selected;

%Run ocular artifact correction
save_cleaned_eeg = eye_movement_correction_function_COGMO( ...
    eeg_to_be_corrected, ...
    eeg_to_be_corrected_data, ...
    vertical_eye_selected, ...
    horizontal_eye_selected,...
    vertical_eye_selected_weights, ...
    horizontal_eye_selected_weights, ...
    data_selected_weights);

%Saving the data
data_exported = eeg_to_be_corrected;
data_exported.data(eeg_idx,:) = save_cleaned_eeg;
% data_exported.data = [];
% data_exported.data =save_cleaned_eeg; 

%reshape cleaned EEG channels back to original format =====
if is_epoched
    % back to channels x timepoints x epochs
    cleaned_3d = reshape(save_cleaned_eeg, numel(eeg_idx), orig_size(2), orig_size(3));
    data_exported.data(eeg_idx,:,:) = cleaned_3d;
else
    data_exported.data(eeg_idx,:) = save_cleaned_eeg;
end

end