function [curves, t_ms, figH, S_prp] = plot_manner_prp_curves_from_EEG(EEG, varargin)
% Plot average PRP curves by manner from an EEGLAB EEG struct that contains:
%   EEG.data                 : [nChan x nSamples x nEpochs]
%   EEG.phoneme_onset_index  : [nPhonemes x nSamples x nEpochs]
%   EEG.phonemes             : [nPhonemes x 1]
%
% This function reconstructs PRP trials on the fly from onset labels, then
% follows the same logic as your original function:
%   1) subset channels
%   2) average repeated instances within each phoneme
%   3) assign manner labels
%   4) average phoneme-level PRPs within each manner
%
% Output:
%   curves : struct of plotted curves by manner
%   t_ms   : time axis in ms
%   figH   : figure handle
%   S_prp  : reconstructed PRP-style struct for debugging/reuse
%
% Example:
%   [curves, t_ms, figH, S_prp] = plot_manner_prp_curves_from_EEG(EEG, ...
%       'Channels', {'Fz','FC1','FC2'}, ...
%       'BaselineDuration', 0.1, ...
%       'PRPDuration', 0.5, ...
%       'Legends', true);

%% defaults
default_selected_manner = {'vowel','stop','fricative','nasal_approximate'};
default_manner_colors = containers.Map( ...
    {'vowel','stop','fricative','nasal_approximate'}, ...
    {[1 0 0], [0 0 1], [0.5 0 0.5], [1 0.67 0]} ...
);

%% parse input
p = inputParser;
p.FunctionName = 'plot_manner_prp_curves_from_EEG';

p.addRequired('EEG', @(x) isstruct(x) && isscalar(x));
p.addParameter('Channels', [], @(x) isempty(x) || iscellstr(x) || isstring(x));
p.addParameter('Manners', default_selected_manner, ...
    @(x) isempty(x) || iscellstr(x) || isstring(x));
p.addParameter('Phoneme2Manner', [], ...
    @(x) isempty(x) || isa(x, 'containers.Map'));
p.addParameter('MannerColors', default_manner_colors, ...
    @(x) isa(x, 'containers.Map'));
p.addParameter('Legends', false, ...
    @(x) isscalar(x) && (islogical(x) || (isnumeric(x) && any(x == [0 1]))));
p.addParameter('xlim', [], @(x) isempty(x) || (isnumeric(x) && numel(x) == 2));
p.addParameter('ylim', [], @(x) isempty(x) || (isnumeric(x) && numel(x) == 2));

% required to reconstruct PRPs
p.addParameter('BaselineDuration', 0.1, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('PRPDuration', 0.5, @(x) isnumeric(x) && isscalar(x) && x > 0);

p.parse(EEG, varargin{:});

selected_ch     = p.Results.Channels;
selected_manner = cellstr(string(p.Results.Manners));
phoneme2manner  = p.Results.Phoneme2Manner;
manner_colors   = p.Results.MannerColors;
showLegend      = logical(p.Results.Legends);
xLimUser        = p.Results.xlim;
yLimUser        = p.Results.ylim;
baseline_duration = p.Results.BaselineDuration;
prp_duration      = p.Results.PRPDuration;

%% sanity check on EEG struct
required_fields = {'data', 'phoneme_onset_index', 'phonemes'};
for i = 1:numel(required_fields)
    if ~isfield(EEG, required_fields{i})
        error('EEG must contain field "%s".', required_fields{i});
    end
end

if ndims(EEG.data) ~= 3
    error('EEG.data must have size [nChan x nSamples x nEpochs].');
end

if ndims(EEG.phoneme_onset_index) ~= 3
    error('EEG.phoneme_onset_index must have size [nPhonemes x nSamples x nEpochs].');
end

phonemes = local_to_cellstr(EEG.phonemes);
nPh = numel(phonemes);

[nChan, nSamples, nEpochs] = size(EEG.data);
[mask_nPh, mask_nSamples, mask_nEpochs] = size(EEG.phoneme_onset_index);

if mask_nPh ~= nPh
    error('Mismatch: numel(EEG.phonemes) = %d, but size(EEG.phoneme_onset_index,1) = %d.', ...
        nPh, mask_nPh);
end
if mask_nSamples ~= nSamples || mask_nEpochs ~= nEpochs
    error(['Mismatch between EEG.data [%d x %d x %d] and EEG.phoneme_onset_index [%d x %d x %d].'], ...
        nChan, nSamples, nEpochs, mask_nPh, mask_nSamples, mask_nEpochs);
end

if isfield(EEG, 'srate') && ~isempty(EEG.srate)
    sr = EEG.srate;
elseif isfield(EEG, 'sr') && ~isempty(EEG.sr)
    sr = EEG.sr;
else
    error('EEG must contain sampling rate in EEG.srate or EEG.sr.');
end

%% reconstruct PRP trials from onset masks
baseline_samples = round(baseline_duration * sr);
prp_duration_samples = round(prp_duration * sr);
window_samples = baseline_samples + prp_duration_samples;

% will collect:
% prp_eegs     : [nPRP x nChan x nTime]
% prp_phonemes : [nPRP x 1]
prp_eegs = [];
prp_phonemes = {};

skip_begin = 0;
skip_end = 0;

for ep = 1:nEpochs
    epoch_data = EEG.data(:, :, ep);  % [nChan x nSamples]

    for ph = 1:nPh
        onset_idx = find(squeeze(EEG.phoneme_onset_index(ph, :, ep)) > 0);

        if isempty(onset_idx)
            continue
        end

        for k = 1:numel(onset_idx)
            on = onset_idx(k);

            % same PRP-style window as your old workflow:
            % baseline before onset + response after onset
            win_start = on - baseline_samples;
            win_end   = win_start + window_samples - 1;

            if win_start < 1
                skip_begin = skip_begin + 1;
                continue
            end

            if win_end > nSamples
                skip_end = skip_end + 1;
                continue
            end

            seg = epoch_data(:, win_start:win_end);   % [nChan x nTime]

            if isempty(prp_eegs)
                prp_eegs = permute(seg, [3 1 2]); % initialize as [1 x nChan x nTime]
            else
                prp_eegs(end+1, :, :) = seg; %#ok<AGROW>
            end

            prp_phonemes{end+1, 1} = phonemes{ph}; %#ok<AGROW>
        end
    end
end

if isempty(prp_eegs)
    error('No valid PRP windows could be reconstructed from EEG.phoneme_onset_index.');
end

%% build PRP-style struct to match original logic
S_prp = struct();
S_prp.eegs = prp_eegs;                 % [nPRP x nChan x nTime]
S_prp.phonemes = prp_phonemes(:);      % [nPRP x 1]
S_prp.chan_labels = local_chanlabels(EEG);
S_prp.sr = sr;
S_prp.baseline_duration = baseline_duration;
S_prp.prp_duration = prp_duration;
S_prp.skip_begin = skip_begin;
S_prp.skip_end = skip_end;

%% subset channels
[S_ch, ~] = subset_by_channels(S_prp, selected_ch);
eegs_ch = S_ch.eegs;   % [nPRP x nSelChan x nT]

nSelChan = size(eegs_ch, 2);
if nSelChan == 0
    error('No channels selected. Check selected_ch names vs EEG.chanlocs / chan_labels.');
end

%% get phoneme labels from reconstructed PRP rows
phoneme_rows = local_to_cellstr(S_ch.phonemes);

if numel(phoneme_rows) ~= size(eegs_ch, 1)
    error('Number of phoneme labels (%d) does not match number of PRP rows in eegs (%d).', ...
        numel(phoneme_rows), size(eegs_ch, 1));
end

%% build one PRP per unique phoneme
unique_phonemes = unique(phoneme_rows, 'stable');

nUniquePh = numel(unique_phonemes);
nT = size(eegs_ch, 3);

phoneme_prps = nan(nUniquePh, nSelChan, nT);
phoneme_labels = cell(nUniquePh, 1);

for i = 1:nUniquePh
    ph = unique_phonemes{i};
    idx = strcmpi(phoneme_rows, ph);

    if ~any(idx)
        continue
    end

    % average repeated instances of the same phoneme
    phoneme_prps(i, :, :) = mean(eegs_ch(idx, :, :), 1);
    phoneme_labels{i} = ph;
end

%% assign manner labels
tmpStruct = struct();
tmpStruct.phonemes = phoneme_labels;

if isempty(phoneme2manner)
    [full_manners, valid_manners, validMask] = assign_manner_labels(tmpStruct);
else
    [full_manners, valid_manners, validMask] = assign_manner_labels( ...
        tmpStruct, 'Phoneme2Manner', phoneme2manner);
end

full_manners = full_manners(:);
validMask = validMask(:);

%% decide which manners to plot
unique_manner = unique(cellstr(string(valid_manners)));
selected_manner = intersect(selected_manner, unique_manner, 'stable');

if isempty(selected_manner)
    error('None of the requested manners exists after mapping phonemes to manners.');
end

%% time axis
t_ms = ((0:nT-1) ./ sr - baseline_duration) * 1000;

%% plot curves
curves = struct();
figH = figure('Color', 'w');
axes('NextPlot', 'add');
title('PRP average curves by manner');
xlabel('Time (ms)');
ylabel('Amplitude (\muV)');

for m = 1:numel(selected_manner)
    key = selected_manner{m};

    idxPh = validMask & strcmpi(full_manners, key);

    if ~any(idxPh)
        warning('No phoneme-level PRPs found for manner "%s". Skipping.', key);
        continue
    end

    manner_prps = phoneme_prps(idxPh, :, :);   % [nPhInManner x nSelChan x nT]

    % average across phoneme cases first, then across channels
    tmp = squeeze(mean(manner_prps, 1));   % [nSelChan x nT]
    y = mean(tmp, 1);                      % [1 x nT]
    y = y(:)';

    safeKey = matlab.lang.makeValidName(key);
    curves.(safeKey) = y;

    if isKey(manner_colors, key)
        plot(t_ms, y, 'LineWidth', 2, ...
            'DisplayName', key, 'Color', manner_colors(key));
    else
        plot(t_ms, y, 'LineWidth', 2, 'DisplayName', key);
    end
end

xline(0, '--', 'LineWidth', 1.5, 'DisplayName', 'Onset');

if showLegend
    legend('Location', 'best', 'Interpreter', 'none');
end
if ~isempty(xLimUser)
    xlim(xLimUser);
end
if ~isempty(yLimUser)
    ylim(yLimUser);
end

grid on;
hold off;

fprintf('Reconstructed PRPs: %d\n', size(S_prp.eegs, 1));
fprintf('Skipped at beginning: %d\n', skip_begin);
fprintf('Skipped at end: %d\n', skip_end);

end


%% ---------- helper functions ----------
function out = local_to_cellstr(x)
% Convert common MATLAB text containers to cellstr column

    if iscell(x)
        out = cellfun(@char, x, 'UniformOutput', false);
    elseif isstring(x)
        out = cellstr(x);
    elseif ischar(x)
        out = cellstr(x);
    else
        error('Unsupported text container type: %s', class(x));
    end

    out = out(:);
end

function chan_labels = local_chanlabels(EEG)
% Get channel labels as cellstr column

    if isfield(EEG, 'chan_labels') && ~isempty(EEG.chan_labels)
        chan_labels = cellstr(string(EEG.chan_labels(:)));
    elseif isfield(EEG, 'chanlocs') && ~isempty(EEG.chanlocs)
        chan_labels = arrayfun(@(x) string(x.labels), EEG.chanlocs, 'UniformOutput', false);
        chan_labels = cellstr(string(chan_labels(:)));
    else
        error('Could not find channel labels in EEG.chan_labels or EEG.chanlocs.');
    end
end

function [full_manners, valid_manners, valid_manners_idx] = assign_manner_labels(inputdata, varargin)

default_phoneme2manner = containers.Map( ...
    {'AA','AE','AO','AW','AY','EH','ER','EY','IH','IY','OW','UW','AH', ...
     'B','D','G','K','P','T', ...
     'F','S','V','Z','SH','DH','HH','CH', ...
     'M','N','NG','L','R','W'}, ...
    {'vowel','vowel','vowel','vowel','vowel','vowel','vowel','vowel','vowel','vowel','vowel','vowel','vowel', ...
     'stop','stop','stop','stop','stop','stop', ...
     'fricative','fricative','fricative','fricative','fricative','fricative','fricative','fricative', ...
     'nasal_approximate','nasal_approximate','nasal_approximate','nasal_approximate','nasal_approximate','nasal_approximate'} ...
);

p = inputParser;
p.FunctionName = 'assign_manner_labels';
p.addRequired('inputdata', @(x) ...
    (isstruct(x) && isscalar(x) && isfield(x,'phonemes')) || ...
    (iscellstr(x) && ~isempty(x)) );
p.addParameter('Phoneme2Manner', default_phoneme2manner, ...
    @(x) isa(x,'containers.Map'));
p.parse(inputdata, varargin{:});
phoneme2manner = p.Results.Phoneme2Manner;

if isstruct(inputdata)
    phonemes = inputdata.phonemes;
else
    phonemes = inputdata;
end

phonemes = phonemes(:);
nTrial = numel(phonemes);

full_manners = cell(nTrial, 1);
missingPhon = {};

for i = 1:nTrial
    ph = string(phonemes{i});
    ph = strtrim(ph);
    ph = char(ph);

    if isKey(phoneme2manner, ph)
        full_manners{i} = phoneme2manner(ph);
    else
        full_manners{i} = '';
        missingPhon{end+1} = ph; %#ok<AGROW>
    end
end

if ~isempty(missingPhon)
    missingPhon = unique(missingPhon);
    warning('Unknown phonemes not found in dictionary: %s', strjoin(missingPhon, ','));
end

valid_manners_idx = ~cellfun(@isempty, full_manners);
valid_manners = full_manners(valid_manners_idx);

end

function [S_out, chIdx] = subset_by_channels(S, selected_ch)
% Subset S.eegs and S.chan_labels by selected channels.

if isempty(selected_ch)
    chIdx = 1:numel(S.chan_labels);
    S_out = S;
    return
end

selected_ch = cellstr(string(selected_ch));
chan_labels = cellstr(string(S.chan_labels(:)));
[tf, chIdx] = ismember(selected_ch, chan_labels);

if any(~tf)
    missing = selected_ch(~tf);
    warning('Some channels not found in chan_labels: %s', strjoin(missing, ', '));
    chIdx = chIdx(tf);
end

S_out = S;
S_out.chan_labels = S.chan_labels(chIdx);

if isfield(S, 'eegs')
    % eegs: [nTrial x nChan x nT]
    S_out.eegs = S.eegs(:, chIdx, :);
else
    error('S must contain field ''eegs''.');
end

end