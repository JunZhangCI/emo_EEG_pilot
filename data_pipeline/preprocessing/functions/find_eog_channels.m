function [veog_idx, heog_idx] = find_eog_channels(labels)

labels = upper(labels);

veog_idx = find(strcmp(labels, 'VE0G'));   % zero not letter O
heog_idx = find(strcmp(labels, 'HEOG'));

if isempty(veog_idx)
    error('VE0G channel not found!');
end

if isempty(heog_idx)
    error('HEOG channel not found!');
end

fprintf('Detected EOG: VE0G=%d, HEOG=%d\n', veog_idx, heog_idx);
end