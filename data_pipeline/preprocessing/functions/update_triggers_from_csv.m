function EEG = update_triggers_from_csv(csv_dir, EEG_fname, EEG, exclude_event)
% UPDATE_TRIGGERS_FROM_CSV(maindir, EEG_fname, EEG, exclude_event)
%
%   Updates event triggers in an EEGLAB EEG dataset based on corrected trigger
%   latencies stored in a CSV file for the corresponding subject and task.
%   The function also cleans event labels, optionally excludes specified event
%   codes, and returns the updated EEG structure.
%
%   Key features:
%     - Loads a pre-existing EEG structure and checks its validity.
%     - Identifies the subject and task from the EEG filename.
%     - Locates CSV files containing trigger latency adjustments for the subject.
%     - If multiple matching CSV files exist, prompts the user to choose one.
%     - Cleans heterogeneous event labels, converting nested cells or strings
%       to numeric event codes.
%     - Optionally excludes user-defined event codes from processing.
%     - Prompts the user to select event codes to exclude if `exclude_event`
%       is not provided and there are multiple unique events.
%     - Updates EEG.event.latency values according to the CSV `samps_to_add` field.
%     - Returns a valid EEGLAB EEG structure with updated triggers.
%
%   Inputs:
%     maindir       - Base directory of the experiment
%     EEG_fname     - Name of the EEG .set file (used for task and subject identification)
%     EEG           - EEGLAB EEG structure corresponding to EEG_fname
%     exclude_event - (Optional) Numeric vector of event codes to exclude
%
%   Outputs:
%     EEG           - Updated EEGLAB EEG structure with cleaned events and
%                     corrected trigger latencies
%
%   Example usage:
%     EEG = update_triggers_from_csv('C:/EEGProject', 'sub-001_task-da_down.set', EEG, [255 999]);
%
%   Notes:
%     - If `exclude_event` is empty and there are more than 3 unique events,
%       the function will prompt the user to select events to exclude.
%     - Only events present in both the EEG dataset and the CSV file are updated.
%     - The function converts all event types to numeric strings to ensure
%       compatibility with EEGLAB epoching functions.
%     - This function does not save the EEG dataset; it only returns the updated
%       structure.

%% Validate parameters and set defaults
if isempty(EEG)
    error('An EEGLAB EEG struct should be entered')
end

if ~exist('exclude_event','var') || isempty(exclude_event)
    exclude_event = [];
end

fprintf('processing file %s\n', EEG_fname);
namepts = strsplit(EEG_fname,'_');

%% Locate CSV files
csv_files = dir(fullfile(csv_dir, '*.csv'));

if isempty(csv_files)
    error('No CSV files found in %s', csv_dir);
end

% Determine task 
taskfull = namepts{4};
task = erase(taskfull,'task-');

% continuous_tasks = {'alice','mix','einstein'};
% discrete_tasks = {'da', 'click', 'puretone'};
% if any(strcmpi(task, continuous_tasks))
%     task = 'continuous';
% elseif ~any(strcmp(task, discrete_tasks))
%     error('%s \n This file is not part of any valid task.', EEG_fname)
% end

%% Find adjusted trigger time points CSV file in the subject folder
% locate all matching files
valid_csv ={};
k = 1;
for csv = 1: length(csv_files)
    [~, csvname, ~] = fileparts(csv_files(csv).name);
    csv_namepts = strsplit(csvname, '_');
    csv_task = csv_namepts(3);
    if strcmpi(csv_task, task)
        valid_csv{k} = csv_files(csv).name;
        k = k + 1; 
    end
end
% select one file
if isempty(valid_csv)
    error("No matching CSV for task: %s", task);
elseif length(valid_csv) == 1
    selected_file = valid_csv{1};
else
    % Display all found files
    fprintf('Multiple CSV files found:\n');
    for f = 1:length(valid_csv)
        fprintf('%d: %s\n', f, valid_csv{f});
    end
    
    % Ask user to choose
    choice = input(sprintf('Select a file (1-%d): ', length(valid_csv)));
    
    % Validate choice
    if choice < 1 || choice > length(valid_csv)
        error('Invalid selection. Please run again and choose a valid index.');
    end
    selected_file = valid_csv{choice};
end

% Full path to the CSV file
trigtimes_file = fullfile(csv_dir, selected_file);
trigtimes = readtable(trigtimes_file);
fprintf('CSV file selected: %s\n', trigtimes_file)

% Load EEG data
EEG = eeg_checkset(EEG);
fprintf('Loaded %s: %d channels, %.1f Hz\n', EEG_fname, EEG.nbchan, EEG.srate);

% %% debug: time course with updated triggers
% % Extract EEG-only dataset for plotting
% types = {EEG.chanlocs.type};
% eeg_idx = find(strcmpi(types, 'EEG'));        % indices in the full EEG struct that are EEG channels
% 
% % Create EEG-only dataset for plotting (keeps chanlocs consistent)
% if ~isempty(eeg_idx)
%     EEG_only = pop_select(EEG, 'channel', eeg_idx);
% else
%     error('No EEG channels found for plotting.');
% end
% pop_eegplot(EEG_only, 1, 1, 1, [], 'winlength', 60, 'title', 'Debug: triggers before update','spacing',80);
% pause;

%% Event cleanning
% Extract and clean event types
evt_raw = {EEG.event.type};
nEvt = length(evt_raw);
evt_nums = nan(1, nEvt);   % numeric output

for ii = 1:nEvt
    val = evt_raw{ii};

    % --- unwrap nested cells ---
    while iscell(val)
        val = val{1};
    end

    % --- CASE 1: already numeric ---
    if isnumeric(val)
        evt_nums(ii) = val;
        continue
    end

    % --- CASE 2: string → extract numeric part ---
    if ischar(val) || isstring(val)
        % remove everything except digits, minus signs, decimal points
        num_str = regexprep(val, '[^\d.-]', '');

        % convert to number
        evt_nums(ii) = str2double(num_str);
        continue
    end

    % --- CASE 3: anything else → skip ---
    evt_nums(ii) = NaN;
end

% Prompt user to select event to exclude
if isempty(exclude_event) && length(unique(evt_nums)) > 1
    fprintf('Events: %s\n', num2str(unique(evt_nums)));
    ok = input('Exclude trigger? (y/n): ', 's');
    if strcmpi(ok, 'y')
        valid_events = unique(evt_nums);
        valid_input = false;
        while ~valid_input
            user_str = input('Enter event codes to exclude (e.g., 1 2 or 1,2): ', 's');
            exclude_event = str2num(user_str); %#ok<ST2NM>
            if isempty(exclude_event) || ~all(ismember(exclude_event, valid_events))
                fprintf('Invalid event codes entered. Valid codes are: [%s]\n', num2str(valid_events));
            else
                valid_input = true;
            end
        end
    end
end

% Indices of events that match excluded codes
excluded_idx = find(ismember(evt_nums, exclude_event));
fprintf('Excluded event indices in EEG.event: [%s]\n', num2str(excluded_idx));
% Apply exclusion rules
if isempty(exclude_event)
    valid_idx = ~isnan(evt_nums);
else
    valid_idx = ~isnan(evt_nums) & ~ismember(evt_nums, exclude_event);
end

% Keep only valid events
evt_nums = evt_nums(valid_idx);
EEG.event = EEG.event(valid_idx);
% Convert event types to strings (required by pop_epoch)
for ii = 1:length(EEG.event)
    EEG.event(ii).type = num2str(evt_nums(ii));
end

% Final check
if isempty(EEG.event)
    error('No valid events found in %s after filtering.', EEG_fname);
end

fprintf('Events: %s\n', num2str(unique(evt_nums)));

%% Update triggers latencies

% Match current file to CSV filename
rawname = extractBefore(EEG_fname, '_down');   % removes preprocessing suffixes
if isempty(rawname)
    rawname = strjoin(namepts(1:5), '_');
end
% fprintf('Name extracted for matching: %s\n', rawname)
match_idx = contains(trigtimes.filename, rawname);
Tsub = trigtimes(match_idx, :);
% valid_evt_idx = setdiff(1:height(Tsub), excluded_idx);

% Adjust trigger latencies
% disp(['EEG has ' num2str(length(EEG.event)) ' events']);
% disp(['CSV has ' num2str(length(valid_evt_idx)) ' rows']);
for ev = 1:length(EEG.event)
    this_trig = evt_nums(ev);   % trigger code in EEG
    row_idx = find(Tsub.trigger == this_trig, 1, 'first');
    if isempty(row_idx)
        warning('No CSV row found for trigger %d (event %d). Skipping.', ...
                this_trig, ev);
        continue
    end
    samps_to_add = Tsub.samps_to_add(row_idx);
    EEG.event(ev).latency = EEG.event(ev).latency + samps_to_add;
    % remove used row so next identical trigger matches next row
    Tsub(row_idx,:) = [];
    % samps_to_add = Tsub.samps_to_add(valid_evt_idx(ev));
    % EEG.event(ev).latency = EEG.event(ev).latency + 0;
    % EEG.event(ev).latency = EEG.event(ev).latency + samps_to_add;
end

% %% debug: time course with updated triggers
% % Extract EEG-only dataset for plotting
% types = {EEG.chanlocs.type};
% eeg_idx = find(strcmpi(types, 'EEG'));        % indices in the full EEG struct that are EEG channels
% 
% % Create EEG-only dataset for plotting (keeps chanlocs consistent)
% if ~isempty(eeg_idx)
%     EEG_only = pop_select(EEG, 'channel', eeg_idx);
% else
%     error('No EEG channels found for plotting.');
% end
% pop_eegplot(EEG_only, 1, 1, 1, [], 'winlength', 60, 'title', 'Debug: check updated triggers','spacing',80);
% pause;

% Save the updated EEG structure
EEG = eeg_checkset(EEG);
fprintf('Updated %d trigger latencies for %s.\n', length(EEG.event), EEG_fname);
end    